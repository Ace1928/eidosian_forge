from __future__ import annotations
import importlib
import os
from typing import Optional
from transformers import (
from .config import PeftConfig
from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .peft_model import (
from .utils.constants import TOKENIZER_CONFIG_NAME
from .utils.other import check_file_exists_on_hf_hub
class AutoPeftModel(_BaseAutoPeftModel):
    _target_class = None
    _target_peft_class = PeftModel