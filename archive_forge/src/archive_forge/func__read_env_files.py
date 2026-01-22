import os
import warnings
from pathlib import Path
from typing import AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, Union
from .config import BaseConfig, Extra
from .fields import ModelField
from .main import BaseModel
from .types import JsonWrapper
from .typing import StrPath, display_as_type, get_origin, is_union
from .utils import deep_update, lenient_issubclass, path_type, sequence_like
def _read_env_files(self, case_sensitive: bool) -> Dict[str, Optional[str]]:
    env_files = self.env_file
    if env_files is None:
        return {}
    if isinstance(env_files, (str, os.PathLike)):
        env_files = [env_files]
    dotenv_vars = {}
    for env_file in env_files:
        env_path = Path(env_file).expanduser()
        if env_path.is_file():
            dotenv_vars.update(read_env_file(env_path, encoding=self.env_file_encoding, case_sensitive=case_sensitive))
    return dotenv_vars