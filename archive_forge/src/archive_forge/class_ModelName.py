from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools import BaseTool
from langchain_community.tools.steamship_image_generation.utils import make_image_public
class ModelName(str, Enum):
    """Supported Image Models for generation."""
    DALL_E = 'dall-e'
    STABLE_DIFFUSION = 'stable-diffusion'