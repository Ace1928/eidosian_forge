from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_vipllava import VipLlavaConfig

        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, VipLlavaForConditionalGeneration

        >>> model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", device_map="auto", torch_dtype=torch.float16)
        >>> processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

        >>> prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
        >>> question = "Can you please describe this image?"
        >>> prompt = prompt.format(question)
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=text, images=image, return_tensors="pt").to(0, torch.float16)

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=20)
        >>> processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        The image features a brown and white cat sitting on a green surface, with a red ball in its
        ```