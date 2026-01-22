from __future__ import annotations as _annotations
from pathlib import Path
from typing import Any, ClassVar
from pydantic import ConfigDict
from pydantic._internal._config import config_keys
from pydantic._internal._utils import deep_update
from pydantic.main import BaseModel
from .sources import (
def _settings_build_values(self, init_kwargs: dict[str, Any], _case_sensitive: bool | None=None, _env_prefix: str | None=None, _env_file: DotenvType | None=None, _env_file_encoding: str | None=None, _env_ignore_empty: bool | None=None, _env_nested_delimiter: str | None=None, _env_parse_none_str: str | None=None, _secrets_dir: str | Path | None=None) -> dict[str, Any]:
    case_sensitive = _case_sensitive if _case_sensitive is not None else self.model_config.get('case_sensitive')
    env_prefix = _env_prefix if _env_prefix is not None else self.model_config.get('env_prefix')
    env_file = _env_file if _env_file != ENV_FILE_SENTINEL else self.model_config.get('env_file')
    env_file_encoding = _env_file_encoding if _env_file_encoding is not None else self.model_config.get('env_file_encoding')
    env_ignore_empty = _env_ignore_empty if _env_ignore_empty is not None else self.model_config.get('env_ignore_empty')
    env_nested_delimiter = _env_nested_delimiter if _env_nested_delimiter is not None else self.model_config.get('env_nested_delimiter')
    env_parse_none_str = _env_parse_none_str if _env_parse_none_str is not None else self.model_config.get('env_parse_none_str')
    secrets_dir = _secrets_dir if _secrets_dir is not None else self.model_config.get('secrets_dir')
    init_settings = InitSettingsSource(self.__class__, init_kwargs=init_kwargs)
    env_settings = EnvSettingsSource(self.__class__, case_sensitive=case_sensitive, env_prefix=env_prefix, env_nested_delimiter=env_nested_delimiter, env_ignore_empty=env_ignore_empty, env_parse_none_str=env_parse_none_str)
    dotenv_settings = DotEnvSettingsSource(self.__class__, env_file=env_file, env_file_encoding=env_file_encoding, case_sensitive=case_sensitive, env_prefix=env_prefix, env_nested_delimiter=env_nested_delimiter, env_ignore_empty=env_ignore_empty, env_parse_none_str=env_parse_none_str)
    file_secret_settings = SecretsSettingsSource(self.__class__, secrets_dir=secrets_dir, case_sensitive=case_sensitive, env_prefix=env_prefix)
    sources = self.settings_customise_sources(self.__class__, init_settings=init_settings, env_settings=env_settings, dotenv_settings=dotenv_settings, file_secret_settings=file_secret_settings)
    if sources:
        return deep_update(*reversed([source() for source in sources]))
    else:
        return {}