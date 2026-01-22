from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
class ModelConfigBody(BaseModel):
    max_tokens: int = Field(default=None, gt=0, le=102400)
    temperature: float = Field(default=None, ge=0, le=3)
    top_p: float = Field(default=None, ge=0, le=1)
    presence_penalty: float = Field(default=None, ge=-2, le=2)
    frequency_penalty: float = Field(default=None, ge=-2, le=2)
    penalty_decay: float = Field(default=None, ge=0.99, le=0.999)
    top_k: int = Field(default=None, ge=0, le=25)
    global_penalty: bool = Field(default=None, description='When generating a response, whether to include the submitted prompt as a penalty factor. By turning this off, you will get the same generated results as official RWKV Gradio. If you find duplicate results in the generated results, turning this on can help avoid generating duplicates.')
    model_config = {'json_schema_extra': {'example': {'max_tokens': 1000, 'temperature': 1, 'top_p': 0.3, 'presence_penalty': 0, 'frequency_penalty': 1, 'penalty_decay': 0.996, 'global_penalty': False}}}