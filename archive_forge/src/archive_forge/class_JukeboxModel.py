import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
@add_start_docstrings('The bare JUKEBOX Model used for music generation. 4 sampling techniques are supported : `primed_sample`, `upsample`,\n    `continue_sample` and `ancestral_sample`. It does not have a `forward` method as the training is not end to end. If\n    you want to fine-tune the model, it is recommended to use the `JukeboxPrior` class and train each prior\n    individually.\n    ', JUKEBOX_START_DOCSTRING)
class JukeboxModel(JukeboxPreTrainedModel):
    _no_split_modules = ['JukeboxBlock']

    def __init__(self, config):
        super().__init__(config)
        vqvae_config = config.vqvae_config
        self.vqvae = JukeboxVQVAE(vqvae_config)
        self.set_shared_params(config)
        self.priors = nn.ModuleList([JukeboxPrior(config.prior_configs[level], level) for level in range(config.nb_priors)])

    def set_shared_params(self, model_config):
        """
        Initialises the parameters that are shared. This has to be done here because the list of `JukeboxPriorConfig`
        is nest, and is thus unreachable in the `from_dict` function
        """
        for config in model_config.prior_configs:
            config.sampling_rate = model_config.sampling_rate
            config.timing_dims = model_config.timing_dims
            config.min_duration = model_config.min_duration
            config.max_duration = model_config.max_duration
            config.max_nb_genres = model_config.max_nb_genres
            config.metadata_conditioning = model_config.metadata_conditioning

    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.decode(music_tokens, start_level, end_level, bs_chunks)

    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.encode(input_audio, start_level, end_level, bs_chunks)

    def split_batch(self, obj, n_samples, split_size):
        n_passes = (n_samples + split_size - 1) // split_size
        if isinstance(obj, torch.Tensor):
            return torch.split(obj, split_size, dim=0)
        elif isinstance(obj, list):
            return list(zip(*[torch.split(item, split_size, dim=0) for item in obj]))
        elif obj is None:
            return [None] * n_passes
        else:
            raise TypeError('Unknown input type')

    def sample_partial_window(self, music_tokens, labels, offset, sampling_kwargs, level, tokens_to_sample, max_batch_size):
        prior = self.priors[level]
        sampled_tokens = music_tokens[level]
        n_ctx = prior.n_ctx
        nb_sampled_tokens = sampled_tokens.shape[1]
        if nb_sampled_tokens < n_ctx - tokens_to_sample:
            sampling_kwargs['sample_tokens'] = nb_sampled_tokens + tokens_to_sample
            start = 0
        else:
            sampling_kwargs['sample_tokens'] = n_ctx
            start = nb_sampled_tokens - n_ctx + tokens_to_sample
        return self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)

    def sample_single_window(self, music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size):
        prior = self.priors[level]
        n_samples = music_tokens[0].shape[0]
        n_ctx = prior.n_ctx
        end = start + n_ctx
        previous_sampled_tokens = music_tokens[level][:, start:end]
        sample_tokens = sampling_kwargs.get('sample_tokens', None)
        if 'sample_tokens' in sampling_kwargs:
            sample_tokens = end - start
        conditioning_tokens = previous_sampled_tokens.shape[1]
        new_tokens = sample_tokens - previous_sampled_tokens.shape[1]
        logger.info(f'Sampling {sample_tokens} tokens for [{start},{start + sample_tokens}]. Conditioning on {conditioning_tokens} tokens')
        if new_tokens <= 0:
            return music_tokens
        music_tokens_conds = prior.get_music_tokens_conds(music_tokens, start, end)
        metadata = prior.get_metadata(labels, start, self.total_length, offset)
        music_tokens_list = self.split_batch(previous_sampled_tokens, n_samples, max_batch_size)
        music_tokens_conds_list = self.split_batch(music_tokens_conds, n_samples, max_batch_size)
        metadata_list = self.split_batch(metadata, n_samples, max_batch_size)
        tokens = []
        iterator = tqdm(zip(music_tokens_list, music_tokens_conds_list, metadata_list), leave=False)
        for music_tokens_i, music_tokens_conds_i, metadata_i in iterator:
            name = ['Ancestral', 'Primed'][music_tokens_i.shape[1] == 0]
            iterator.set_description(f'[prior level {level}] {name} Sampling {sample_tokens} tokens out of {self.total_length // prior.raw_to_tokens}', refresh=True)
            tokens_i = prior.sample(n_samples=music_tokens_i.shape[0], music_tokens=music_tokens_i, music_tokens_conds=music_tokens_conds_i, metadata=metadata_i, **sampling_kwargs)
            tokens.append(tokens_i)
        sampled_tokens = torch.cat(tokens, dim=0)
        music_tokens_new = sampled_tokens[:, -new_tokens:]
        music_tokens[level] = torch.cat([music_tokens[level], music_tokens_new], dim=1)
        return music_tokens

    def sample_level(self, music_tokens, labels, offset, sampling_kwargs, level, total_length, hop_length, max_batch_size):
        if total_length >= self.priors[level].n_ctx:
            iterator = get_starts(total_length, self.priors[level].n_ctx, hop_length)
            for start in iterator:
                music_tokens = self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)
        else:
            music_tokens = self.sample_partial_window(music_tokens, labels, offset, sampling_kwargs, level, total_length, max_batch_size)
        return music_tokens

    @torch.no_grad()
    def _sample(self, music_tokens, labels, sample_levels, metas=None, chunk_size=32, sampling_temperature=0.98, lower_batch_size=16, max_batch_size=16, sample_length_in_seconds=24, compute_alignments=False, sample_tokens=None, offset=0, save_results=True, sample_length=None) -> List[torch.LongTensor]:
        """
        Core sampling function used to generate music tokens. Iterates over the provided list of levels, while saving
        the generated raw audio at each step.

        Args:
            music_tokens (`List[torch.LongTensor]`):
                A sequence of music tokens of length `self.levels` which will be used as context to continue the
                sampling process. Should have `self.levels` tensors, each corresponding to the generation at a certain
                level.
            labels (`List[torch.LongTensor]`):
                List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +
                lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens
                which are used to condition the generation.
            sample_levels (`List[int]`):
                List of the desired levels at which the sampling will be done. A level is equivalent to the index of
                the prior in the list of priors
            metas (`List[Any]`, *optional*):
                Metadatas used to generate the `labels`
            chunk_size (`int`, *optional*, defaults to 32):
                Size of a chunk of audio, used to fill up the memory in chuncks to prevent OOM erros. Bigger chunks
                means faster memory filling but more consumption.
            sampling_temperature (`float`, *optional*, defaults to 0.98):
                Temperature used to ajust the randomness of the sampling.
            lower_batch_size (`int`, *optional*, defaults to 16):
                Maximum batch size for the lower level priors
            max_batch_size (`int`, *optional*, defaults to 16):
                Maximum batch size for the top level priors
            sample_length_in_seconds (`int`, *optional*, defaults to 24):
                Desired length of the generation in seconds
            compute_alignments (`bool`, *optional*, defaults to `False`):
                Whether or not to compute the alignment between the lyrics and the audio using the top_prior
            sample_tokens (`int`, *optional*):
                Precise number of tokens that should be sampled at each level. This is mostly useful for running dummy
                experiments
            offset (`int`, *optional*, defaults to 0):
                Audio offset used as conditioning, corresponds to the starting sample in the music. If the offset is
                greater than 0, the lyrics will be shifted take that intoaccount
            save_results (`bool`, *optional*, defaults to `True`):
                Whether or not to save the intermediate results. If `True`, will generate a folder named with the start
                time.
            sample_length (`int`, *optional*):
                Desired length of the generation in samples.

        Returns: torch.Tensor

        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed
        >>> import torch

        >>> metas = dict(artist="Zac Brown Band", genres="Country", lyrics="I met a traveller from an antique land")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()

        >>> labels = tokenizer(**metas)["input_ids"]
        >>> set_seed(0)
        >>> zs = [torch.zeros(1, 0, dtype=torch.long) for _ in range(3)]
        >>> zs = model._sample(zs, labels, [0], sample_length=40 * model.priors[0].raw_to_tokens, save_results=False)
        >>> zs[0]
        tensor([[1853, 1369, 1150, 1869, 1379, 1789,  519,  710, 1306, 1100, 1229,  519,
              353, 1306, 1379, 1053,  519,  653, 1631, 1467, 1229, 1229,   10, 1647,
             1254, 1229, 1306, 1528, 1789,  216, 1631, 1434,  653,  475, 1150, 1528,
             1804,  541, 1804, 1434]])
        ```
        """
        top_prior = self.priors[0]
        if sample_length is not None:
            total_length = sample_length
        else:
            total_length = int(sample_length_in_seconds * self.config.sampling_rate) // top_prior.raw_to_tokens * top_prior.raw_to_tokens
        if sample_levels is None:
            sample_levels = range(len(self.priors))
        self.total_length = total_length
        for level in sample_levels:
            sampling_kwargs = {'temp': 0.99 if level == len(self.priors) - 1 else sampling_temperature, 'chunk_size': chunk_size, 'sample_tokens': sample_tokens}
            total_token_to_sample = total_length // self.priors[level].raw_to_tokens
            hop_length = int(self.config.hop_fraction[level] * self.priors[level].n_ctx)
            max_batch_size = lower_batch_size if level != sample_levels else max_batch_size
            music_tokens = self.sample_level(music_tokens, labels[level], offset, sampling_kwargs, level, total_token_to_sample, hop_length, max_batch_size)
            if save_results:
                self.vqvae.to(music_tokens[level].device)
                with torch.no_grad():
                    start_level = len(self.priors) - level - 1
                    raw_audio = self.vqvae.decode(music_tokens[:level + 1], start_level=start_level, bs_chunks=music_tokens[level].shape[0])
                logdir = f'jukebox/level_{level}'
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                save_temp_audio(logdir, level, metas=metas, aud=raw_audio.float())
                if compute_alignments and self.priors[0] is not None and (self.priors[0].nb_relevant_lyric_tokens > 0):
                    with torch.no_grad():
                        alignments = get_alignment(music_tokens, labels[0], self.priors[0], self.config)
                    torch.save({'alignments': alignments}, f'{logdir}/lyric_alignments.pt')
        return music_tokens

    @add_start_docstrings('\n        Generates music tokens based on the provided `labels. Will start at the desired prior level and automatically\n        upsample the sequence. If you want to create the audio, you should call `model.decode(tokens)`, which will use\n        the VQ-VAE decoder to convert the music tokens to raw audio.\n\n        Args:\n            labels (`List[torch.LongTensor]`) :\n                List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +\n                lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens\n                which are used to condition the generation.\n            n_samples (`int`, *optional*, default to 1) :\n                Number of samples to be generated in parallel.\n        ')
    def ancestral_sample(self, labels, n_samples=1, **sampling_kwargs) -> List[torch.LongTensor]:
        """
        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed

        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")

        >>> lyrics = "Hey, are you awake? Can you talk to me?"
        >>> artist = "Zac Brown Band"
        >>> genre = "Country"
        >>> metas = tokenizer(artist=artist, genres=genre, lyrics=lyrics)
        >>> set_seed(0)
        >>> music_tokens = model.ancestral_sample(metas.input_ids, sample_length=400)

        >>> with torch.no_grad():
        ...     model.decode(music_tokens)[:, :10].squeeze(-1)
        tensor([[-0.0219, -0.0679, -0.1050, -0.1203, -0.1271, -0.0936, -0.0396, -0.0405,
            -0.0818, -0.0697]])
        ```
        """
        sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors))))
        music_tokens = [torch.zeros(n_samples, 0, dtype=torch.long, device=labels[0].device) for _ in range(len(self.priors))]
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens

    @add_start_docstrings('Generates a continuation of the previously generated tokens.\n\n        Args:\n            music_tokens (`List[torch.LongTensor]` of length `self.levels` ) :\n                A sequence of music tokens which will be used as context to continue the sampling process. Should have\n                `self.levels` tensors, each corresponding to the generation at a certain level.\n        ', JUKEBOX_SAMPLING_INPUT_DOCSTRING)
    def continue_sample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors))))
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens

    @add_start_docstrings('Upsamples a sequence of music tokens using the prior at level `level`.\n\n        Args:\n            music_tokens (`List[torch.LongTensor]` of length `self.levels` ) :\n                A sequence of music tokens which will be used as context to continue the sampling process. Should have\n                `self.levels` tensors, each corresponding to the generation at a certain level.\n        ', JUKEBOX_SAMPLING_INPUT_DOCSTRING)
    def upsample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors) - 1)))
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens

    @add_start_docstrings('Generate a raw audio conditioned on the provided `raw_audio` which is used as conditioning at each of the\n        generation levels. The audio is encoded to music tokens using the 3 levels of the VQ-VAE. These tokens are\n        used: as conditioning for each level, which means that no ancestral sampling is required.\n\n        Args:\n            raw_audio (`List[torch.Tensor]` of length `n_samples` ) :\n                A list of raw audio that will be used as conditioning information for each samples that will be\n                generated.\n        ', JUKEBOX_SAMPLING_INPUT_DOCSTRING)
    def primed_sample(self, raw_audio, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors))))
        self.vqvae.to(raw_audio.device).float()
        with torch.no_grad():
            music_tokens = self.vqvae.encode(raw_audio, start_level=0, end_level=len(self.priors), bs_chunks=raw_audio.shape[0])
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens