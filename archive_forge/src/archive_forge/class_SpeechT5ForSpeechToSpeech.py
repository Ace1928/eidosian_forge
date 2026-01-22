import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
@add_start_docstrings('SpeechT5 Model with a speech encoder and a speech decoder.', SPEECHT5_START_DOCSTRING)
class SpeechT5ForSpeechToSpeech(SpeechT5PreTrainedModel):

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, speech_encoder, speech_decoder)
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)
        self.post_init()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_input_values: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, speaker_embeddings: Optional[torch.FloatTensor]=None, labels: Optional[torch.FloatTensor]=None, stop_labels: Optional[torch.Tensor]=None) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        """
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into `input_values`, the [`SpeechT5Processor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`SpeechT5Processor.__call__`] for details.
        decoder_input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`):
            Float values of input mel spectrogram.

            SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
            `past_key_values`).
        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.
        labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*):
            Float values of target mel spectrogram. Spectrograms can be obtained using [`SpeechT5Processor`]. See
            [`SpeechT5Processor.__call__`] for details.

        Returns:

        Example:

        ```python
        >>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
        >>> from datasets import load_dataset
        >>> import torch

        >>> dataset = load_dataset(
        ...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
        ... )  # doctest: +IGNORE_RESULT
        >>> dataset = dataset.sort("id")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        >>> model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        >>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        >>> # audio file is decoded on the fly
        >>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

        >>> speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file

        >>> set_seed(555)  # make deterministic

        >>> # generate speech
        >>> speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
        >>> speech.shape
        torch.Size([77824])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values = shift_spectrograms_right(labels, self.config.reduction_factor)
        outputs = self.speecht5(input_values=input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_values, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, use_cache=use_cache, speaker_embeddings=speaker_embeddings, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        _, spectrogram, logits = self.speech_decoder_postnet(outputs[0])
        loss = None
        if not return_dict:
            output = (spectrogram,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Seq2SeqSpectrogramOutput(loss=loss, spectrogram=spectrogram, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    @torch.no_grad()
    def generate_speech(self, input_values: torch.FloatTensor, speaker_embeddings: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, threshold: float=0.5, minlenratio: float=0.0, maxlenratio: float=20.0, vocoder: Optional[nn.Module]=None, output_cross_attentions: bool=False, return_output_lengths: bool=False) -> torch.FloatTensor:
        """
        Converts a raw speech waveform into a sequence of mel spectrograms, which are subsequently turned back into a
        speech waveform using a vocoder.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform.

                Values can be obtained by loading a *.flac* or *.wav* audio file into an array of type `List[float]` or
                a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install soundfile*). To prepare the array
                into `input_values`, the [`SpeechT5Processor`] should be used for padding and conversion into a tensor
                of type `torch.FloatTensor`. See [`SpeechT5Processor.__call__`] for details.
            speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.
            vocoder (`nn.Module`, *optional*, defaults to `None`):
                The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
                spectrogram.
            output_cross_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of the decoder's cross-attention layers.
            return_output_lengths (`bool`, *optional*, defaults to `False`):
                Whether or not to return the concrete spectrogram/waveform lengths.

        Returns:
            `tuple(torch.FloatTensor)` comprising various elements depending on the inputs:
            - when `return_output_lengths` is False
                - **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
                - **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(num_frames,)` -- The predicted speech waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
            - when `return_output_lengths` is True
                - **spectrograms** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrograms that
                are padded to the maximum length.
                - **spectrogram_lengths** (*optional*, returned when no `vocoder` is provided) `List[Int]` -- A list of
                all the concrete lengths for each spectrogram.
                - **waveforms** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, num_frames)` -- The predicted speech waveforms that are padded to the maximum length.
                - **waveform_lengths** (*optional*, returned when a `vocoder` is provided) `List[Int]` -- A list of all
                the concrete lengths for each waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
        """
        if speaker_embeddings is None:
            speaker_embeddings = torch.zeros((1, 512), device=input_values.device)
        return _generate_speech(self, input_values, speaker_embeddings, attention_mask, threshold, minlenratio, maxlenratio, vocoder, output_cross_attentions, return_output_lengths)