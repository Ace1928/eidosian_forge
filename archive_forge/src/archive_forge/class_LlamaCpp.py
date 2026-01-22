from typing import TYPE_CHECKING, List, Optional, Union
import torch
from outlines.integrations.llamacpp import (  # noqa: F401
class LlamaCpp:
    """Represents a `llama_cpp` model."""

    def __init__(self, model: 'Llama'):
        self.model = model
        self.tokenizer = LlamaCppTokenizer(model=model)