from __future__ import annotations as _annotations
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple, Union, cast
from dotenv import dotenv_values
from pydantic import AliasChoices, AliasPath, BaseModel, Json
from pydantic._internal._typing_extra import WithArgsTypes, origin_is_union
from pydantic._internal._utils import deep_update, is_model_class, lenient_issubclass
from pydantic.fields import FieldInfo
from typing_extensions import get_args, get_origin
from pydantic_settings.utils import path_type_label
def _replace_env_none_type_values(self, field_value: dict[str, Any]) -> dict[str, Any]:
    """
        Recursively parse values that are of "None" type(EnvNoneType) to `None` type(None).
        """
    values: dict[str, Any] = {}
    for key, value in field_value.items():
        if not isinstance(value, EnvNoneType):
            values[key] = value if not isinstance(value, dict) else self._replace_env_none_type_values(value)
        else:
            values[key] = None
    return values