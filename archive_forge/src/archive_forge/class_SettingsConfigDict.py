from __future__ import annotations as _annotations
from pathlib import Path
from typing import Any, ClassVar
from pydantic import ConfigDict
from pydantic._internal._config import config_keys
from pydantic._internal._utils import deep_update
from pydantic.main import BaseModel
from .sources import (
class SettingsConfigDict(ConfigDict, total=False):
    case_sensitive: bool
    env_prefix: str
    env_file: DotenvType | None
    env_file_encoding: str | None
    env_ignore_empty: bool
    env_nested_delimiter: str | None
    env_parse_none_str: str | None
    secrets_dir: str | Path | None
    json_file: PathType | None
    json_file_encoding: str | None
    yaml_file: PathType | None
    yaml_file_encoding: str | None
    toml_file: PathType | None