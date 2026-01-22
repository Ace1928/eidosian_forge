import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def _env_as_bool(var: str, default: Optional[str]=None, env: Optional[Env]=None) -> bool:
    if env is None:
        env = os.environ
    val = env.get(var, default)
    try:
        val = bool(strtobool(val))
    except (AttributeError, ValueError):
        pass
    return val if isinstance(val, bool) else False