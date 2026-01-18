from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig

        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a (per-pixel) classification loss is computed
            (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs, training=False)
        >>> # logits are of shape (batch_size, num_labels, height/4, width/4)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```