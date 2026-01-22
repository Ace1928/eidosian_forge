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
def _replace_field_names_case_insensitively(self, field: FieldInfo, field_values: dict[str, Any]) -> dict[str, Any]:
    """
        Replace field names in values dict by looking in models fields insensitively.

        By having the following models:

            ```py
            class SubSubSub(BaseModel):
                VaL3: str

            class SubSub(BaseModel):
                Val2: str
                SUB_sub_SuB: SubSubSub

            class Sub(BaseModel):
                VAL1: str
                SUB_sub: SubSub

            class Settings(BaseSettings):
                nested: Sub

                model_config = SettingsConfigDict(env_nested_delimiter='__')
            ```

        Then:
            _replace_field_names_case_insensitively(
                field,
                {"val1": "v1", "sub_SUB": {"VAL2": "v2", "sub_SUB_sUb": {"vAl3": "v3"}}}
            )
            Returns {'VAL1': 'v1', 'SUB_sub': {'Val2': 'v2', 'SUB_sub_SuB': {'VaL3': 'v3'}}}
        """
    values: dict[str, Any] = {}
    for name, value in field_values.items():
        sub_model_field: FieldInfo | None = None
        if not field.annotation or not hasattr(field.annotation, 'model_fields'):
            values[name] = value
            continue
        for sub_model_field_name, f in field.annotation.model_fields.items():
            if not f.validation_alias and sub_model_field_name.lower() == name.lower():
                sub_model_field = f
                break
        if not sub_model_field:
            values[name] = value
            continue
        if lenient_issubclass(sub_model_field.annotation, BaseModel) and isinstance(value, dict):
            values[sub_model_field_name] = self._replace_field_names_case_insensitively(sub_model_field, value)
        else:
            values[sub_model_field_name] = value
    return values