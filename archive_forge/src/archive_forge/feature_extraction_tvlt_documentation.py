from math import ceil
from typing import List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ...utils import TensorType, logging

        Main method to prepare one or several audio(s) for the model.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            return_attention_mask (`bool`, *optional*, default to `True`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default. [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For TvltTransformer models, `attention_mask` should alwys be passed for batched inference, to avoid
                subtle bugs.

                </Tip>

            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline. Current model supports sampling rate 16000 and 44100.
            resample (`bool`, *optional*, defaults to `False`):
                If the sampling rate is not matched, resample the input audio to match.
            mask_audio (`bool`, *optional*, defaults to `False`):
                Whether or not to mask input audio for MAE task.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **audio_values** -- Audio values to be fed to a model, of shape (batch_size, num_channels, height,
              width).

            - **audio_mask** -- Audio masks to be fed to a model, of shape (batch_size, num_audio_patches).
        