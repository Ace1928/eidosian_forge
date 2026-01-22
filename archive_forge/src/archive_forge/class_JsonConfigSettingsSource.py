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
class JsonConfigSettingsSource(InitSettingsSource, ConfigFileSourceMixin):
    """
    A source class that loads variables from a JSON file
    """

    def __init__(self, settings_cls: type[BaseSettings], json_file: PathType | None=DEFAULT_PATH, json_file_encoding: str | None=None):
        self.json_file_path = json_file if json_file != DEFAULT_PATH else settings_cls.model_config.get('json_file')
        self.json_file_encoding = json_file_encoding if json_file_encoding is not None else settings_cls.model_config.get('json_file_encoding')
        self.json_data = self._read_files(self.json_file_path)
        super().__init__(settings_cls, self.json_data)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding=self.json_file_encoding) as json_file:
            return json.load(json_file)