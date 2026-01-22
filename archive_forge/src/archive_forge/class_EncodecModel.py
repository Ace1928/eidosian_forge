import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
@add_start_docstrings('The EnCodec neural audio codec model.', ENCODEC_START_DOCSTRING)
class EncodecModel(EncodecPreTrainedModel):

    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        self.config = config
        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)
        self.quantizer = EncodecResidualVectorQuantizer(config)
        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2 ** self.bits_per_codebook != self.config.codebook_size:
            raise ValueError('The codebook_size must be a power of 2.')
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _encode_frame(self, input_values: torch.Tensor, bandwidth: float, padding_mask: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. If `config.normalize` is set to `True` the input is first
        normalized. The padding mask is required to compute the correct scale.
        """
        length = input_values.shape[-1]
        duration = length / self.config.sampling_rate
        if self.config.chunk_length_s is not None and duration > 1e-05 + self.config.chunk_length_s:
            raise RuntimeError(f'Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}')
        scale = None
        if self.config.normalize:
            input_values = input_values * padding_mask
            mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-08
            input_values = input_values / scale
        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, bandwidth)
        codes = codes.transpose(0, 1)
        return (codes, scale)

    def encode(self, input_values: torch.Tensor, padding_mask: torch.Tensor=None, bandwidth: Optional[float]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], EncodecEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
                bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented
                as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. Select one of {self.config.target_bandwidths}.")
        _, channels, input_length = input_values.shape
        if channels < 1 or channels > 2:
            raise ValueError(f'Number of audio channels must be 1 or 2, but got {channels}')
        chunk_length = self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()
        encoded_frames = []
        scales = []
        step = chunk_length - stride
        if input_length % stride - step != 0:
            raise ValueError('The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly.')
        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset:offset + chunk_length].bool()
            frame = input_values[:, :, offset:offset + chunk_length]
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)
            encoded_frames.append(encoded_frame)
            scales.append(scale)
        encoded_frames = torch.stack(encoded_frames)
        if not return_dict:
            return (encoded_frames, scales)
        return EncodecEncoderOutput(encoded_frames, scales)

    @staticmethod
    def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
        if len(frames) == 0:
            raise ValueError('`frames` cannot be an empty list.')
        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]
        total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]
        frame_length = frames[0].shape[-1]
        time_vec = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()
        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0
        for frame in frames:
            frame_length = frame.shape[-1]
            out[..., offset:offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset:offset + frame_length] += weight[:frame_length]
            offset += stride
        if sum_weight.min() == 0:
            raise ValueError(f'`sum_weight` minimum element must be bigger than zero: {sum_weight}`')
        return out / sum_weight

    def _decode_frame(self, codes: torch.Tensor, scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    def decode(self, audio_codes: torch.Tensor, audio_scales: torch.Tensor, padding_mask: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict or self.config.return_dict
        chunk_length = self.config.chunk_length
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f'Expected one frame, got {len(audio_codes)}')
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []
            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)
            audio_values = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., :padding_mask.shape[-1]]
        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)

    @add_start_docstrings_to_model_forward(ENCODEC_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=EncodecOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: torch.Tensor, padding_mask: Optional[torch.Tensor]=None, bandwidth: Optional[float]=None, audio_codes: Optional[torch.Tensor]=None, audio_scales: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, EncodecModel

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "facebook/encodec_24khz"
        >>> model = EncodecModel.from_pretrained(model_id)
        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict or self.config.return_dict
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()
        if audio_codes is not None and audio_scales is None:
            raise ValueError('You specified `audio_codes` but did not specify the `audio_scales`')
        if audio_scales is not None and audio_codes is None:
            raise ValueError('You specified `audio_scales` but did not specify the `audio_codes`')
        if audio_scales is None and audio_codes is None:
            audio_codes, audio_scales = self.encode(input_values, padding_mask, bandwidth, False)
        audio_values = self.decode(audio_codes, audio_scales, padding_mask, return_dict=return_dict)[0]
        if not return_dict:
            return (audio_codes, audio_values)
        return EncodecOutput(audio_codes=audio_codes, audio_values=audio_values)