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
class PydanticBaseEnvSettingsSource(PydanticBaseSettingsSource):

    def __init__(self, settings_cls: type[BaseSettings], case_sensitive: bool | None=None, env_prefix: str | None=None, env_ignore_empty: bool | None=None, env_parse_none_str: str | None=None) -> None:
        super().__init__(settings_cls)
        self.case_sensitive = case_sensitive if case_sensitive is not None else self.config.get('case_sensitive', False)
        self.env_prefix = env_prefix if env_prefix is not None else self.config.get('env_prefix', '')
        self.env_ignore_empty = env_ignore_empty if env_ignore_empty is not None else self.config.get('env_ignore_empty', False)
        self.env_parse_none_str = env_parse_none_str if env_parse_none_str is not None else self.config.get('env_parse_none_str')

    def _apply_case_sensitive(self, value: str) -> str:
        return value.lower() if not self.case_sensitive else value

    def _extract_field_info(self, field: FieldInfo, field_name: str) -> list[tuple[str, str, bool]]:
        """
        Extracts field info. This info is used to get the value of field from environment variables.

        It returns a list of tuples, each tuple contains:
            * field_key: The key of field that has to be used in model creation.
            * env_name: The environment variable name of the field.
            * value_is_complex: A flag to determine whether the value from environment variable
              is complex and has to be parsed.

        Args:
            field (FieldInfo): The field.
            field_name (str): The field name.

        Returns:
            list[tuple[str, str, bool]]: List of tuples, each tuple contains field_key, env_name, and value_is_complex.
        """
        field_info: list[tuple[str, str, bool]] = []
        if isinstance(field.validation_alias, (AliasChoices, AliasPath)):
            v_alias: str | list[str | int] | list[list[str | int]] | None = field.validation_alias.convert_to_aliases()
        else:
            v_alias = field.validation_alias
        if v_alias:
            if isinstance(v_alias, list):
                for alias in v_alias:
                    if isinstance(alias, str):
                        field_info.append((alias, self._apply_case_sensitive(alias), True if len(alias) > 1 else False))
                    elif isinstance(alias, list):
                        first_arg = cast(str, alias[0])
                        field_info.append((first_arg, self._apply_case_sensitive(first_arg), True if len(alias) > 1 else False))
            else:
                field_info.append((v_alias, self._apply_case_sensitive(v_alias), False))
        elif origin_is_union(get_origin(field.annotation)) and _union_is_complex(field.annotation, field.metadata):
            field_info.append((field_name, self._apply_case_sensitive(self.env_prefix + field_name), True))
        else:
            field_info.append((field_name, self._apply_case_sensitive(self.env_prefix + field_name), False))
        return field_info

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

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field_name, field in self.settings_cls.model_fields.items():
            try:
                field_value, field_key, value_is_complex = self.get_field_value(field, field_name)
            except Exception as e:
                raise SettingsError(f'error getting value for field "{field_name}" from source "{self.__class__.__name__}"') from e
            try:
                field_value = self.prepare_field_value(field_name, field, field_value, value_is_complex)
            except ValueError as e:
                raise SettingsError(f'error parsing value for field "{field_name}" from source "{self.__class__.__name__}"') from e
            if field_value is not None:
                if self.env_parse_none_str is not None:
                    if isinstance(field_value, dict):
                        field_value = self._replace_env_none_type_values(field_value)
                    elif isinstance(field_value, EnvNoneType):
                        field_value = None
                if not self.case_sensitive and lenient_issubclass(field.annotation, BaseModel) and isinstance(field_value, dict):
                    data[field_key] = self._replace_field_names_case_insensitively(field, field_value)
                else:
                    data[field_key] = field_value
        return data