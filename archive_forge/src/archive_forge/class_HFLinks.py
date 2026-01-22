import os
import re
import base64
import requests
import json
import functools
import contextlib
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any, TYPE_CHECKING
from lazyops.utils.logs import logger
from lazyops.types import BaseModel, lazyproperty, Literal
from pydantic.types import ByteSize
class HFLinks(BaseModel):
    repo_id: str
    repo_type: str
    revision: str
    links: List[HFLink] = []

    @lazyproperty
    def is_lora(self):
        return any((link.is_lora for link in self.links))

    @lazyproperty
    def lora_links(self):
        return [link for link in self.links if link.is_lora]

    @lazyproperty
    def pytorch_links(self):
        return [link for link in self.links if link.is_pytorch]

    @lazyproperty
    def pytorch_model_links(self):
        return [link for link in self.links if link.is_pytorch_model]

    @lazyproperty
    def safetensors_links(self):
        return [link for link in self.links if link.is_safetensors]

    @lazyproperty
    def tensorflow_links(self):
        return [link for link in self.links if link.is_tensorflow]

    @lazyproperty
    def config_links(self):
        return [link for link in self.links if link.is_config and (not link.is_tokenizer)]

    @lazyproperty
    def tokenizer_links(self):
        return [link for link in self.links if link.is_tokenizer]

    @lazyproperty
    def text_links(self):
        return [link for link in self.links if link.is_text and (not link.is_tokenizer) and (not link.is_config)]

    @lazyproperty
    def classifications(self) -> List[str]:
        res = []
        if self.lora_links:
            res.append('lora')
        if self.pytorch_model_links or self.pytorch_links:
            res.append('pytorch')
        if self.tensorflow_links:
            res.append('tensorflow')
        if self.safetensors_links:
            res.append('safetensors')
        return res

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx: Union[str, int]) -> Union[List[HFLink], HFLink]:
        """
        Returns the link at the given index or by the type
        """
        if isinstance(idx, int):
            return self.links[idx]
        if hasattr(self, f'{idx}_links'):
            return getattr(self, f'{idx}_links')
        for link in self.links:
            if link.filename == idx:
                return link
        raise KeyError(f'Link with filename `{idx}` not found.')