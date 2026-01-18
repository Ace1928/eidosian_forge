from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig

        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> model = TFSpeech2TextForConditionalGeneration.from_pretrained(
        ...     "facebook/s2t-small-librispeech-asr", from_pt=True
        ... )
        >>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)
        >>> ds.set_format(type="tf")

        >>> input_features = processor(
        ...     ds["speech"][0], sampling_rate=16000, return_tensors="tf"
        ... ).input_features  # Batch size 1
        >>> generated_ids = model.generate(input_features)

        >>> transcription = processor.batch_decode(generated_ids)
        ```