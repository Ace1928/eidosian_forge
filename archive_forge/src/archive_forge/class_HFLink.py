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
class HFLink(BaseModel):
    url: str
    filename: str
    size: ByteSize

    @lazyproperty
    def is_pytorch_model(self):
        return bool(re_patterns['pytorch_model'].match(self.filename))

    @lazyproperty
    def is_safetensors(self):
        return bool(re_patterns['safetensors'].match(self.filename))

    @lazyproperty
    def is_pytorch(self):
        return bool(re_patterns['pytorch'].match(self.filename))

    @lazyproperty
    def is_tensorflow(self):
        return bool(re_patterns['tensorflow'].match(self.filename))

    @lazyproperty
    def is_tokenizer(self):
        return bool(re_patterns['tokenizer'].match(self.filename))

    @lazyproperty
    def is_text(self):
        return bool(re_patterns['text'].match(self.filename))

    @lazyproperty
    def is_lora(self):
        return self.filename.endswith(('adapter_config.json', 'adapter_model.bin'))

    @lazyproperty
    def is_config(self):
        return bool(re_patterns['config'].match(self.filename))