from typing import Any, Callable, List, Optional, Union
import torch
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from ..core import set_seed
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper

        Generate the best of n samples for input queries

        Args:
            tokenized_query (`List[int]` or `torch.Tensor` or `List[torch.Tensor]` or `List[int]`):
                represents either a single tokenized query (a single tensor or a list of integers) or a batch of tokenized queries (a list of tensors or a list of lists of integers)
            skip_special_tokens (`bool`):
                Whether to remove the special tokens from the output
            device (`str` or `torch.device`, *optional*):
                The device on which the model will be loaded
            **generation_kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's `generate` method.
                This is used to override generation config

        Returns:
            List[List[str]]: A list of lists of generated texts
        