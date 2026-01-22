from __future__ import annotations
import pathlib
import secrets
import shutil
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from fastapi import Request
from gradio_client.utils import traverse
from . import wasm_utils
class BaseModelMeta(type(BaseModelV1)):

    def __new__(cls, name, bases, dct):
        if 'model_config' in dct:
            config_class = type('Config', (), {})
            for key, value in dct['model_config'].items():
                setattr(config_class, key, value)
            dct['Config'] = config_class
            del dct['model_config']
        model_class = super().__new__(cls, name, bases, dct)
        return model_class