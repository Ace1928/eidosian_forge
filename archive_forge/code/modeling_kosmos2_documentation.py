import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

        >>> model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
        >>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        >>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "<grounding> An image of"

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> generated_ids = model.generate(
        ...     pixel_values=inputs["pixel_values"],
        ...     input_ids=inputs["input_ids"],
        ...     attention_mask=inputs["attention_mask"],
        ...     image_embeds=None,
        ...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
        ...     use_cache=True,
        ...     max_new_tokens=64,
        ... )
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
        >>> processed_text
        '<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.'

        >>> caption, entities = processor.post_process_generation(generated_text)
        >>> caption
        'An image of a snowman warming himself by a fire.'

        >>> entities
        [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
        ```