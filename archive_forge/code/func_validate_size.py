from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools import BaseTool
from langchain_community.tools.steamship_image_generation.utils import make_image_public
@root_validator(pre=True)
def validate_size(cls, values: Dict) -> Dict:
    if 'size' in values:
        size = values['size']
        model_name = values['model_name']
        if size not in SUPPORTED_IMAGE_SIZES[model_name]:
            raise RuntimeError(f'size {size} is not supported by {model_name}')
    return values