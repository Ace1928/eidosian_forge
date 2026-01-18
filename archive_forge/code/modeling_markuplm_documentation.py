import math
import os
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import (
from ...modeling_utils import (
from ...utils import logging
from .configuration_markuplm import MarkupLMConfig

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForSequenceClassification
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
        >>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/markuplm-base", num_labels=7)

        >>> html_string = "<html> <head> <title>Page Title</title> </head> </html>"
        >>> encoding = processor(html_string, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**encoding)

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```