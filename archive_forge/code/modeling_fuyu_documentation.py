from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModelForCausalLM
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_fuyu import FuyuConfig

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import FuyuProcessor, FuyuForCausalLM
        >>> from PIL import Image
        >>> import requests

        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Generate a coco-style caption.\n"

        >>> inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> generated_ids = model.generate(**model_inputs, max_new_tokens=7)
        >>> generation_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        >>> print(generation_text)
        'A bus parked on the side of a road.'
        ```