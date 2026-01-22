import os
import yaml
import typer
import contextlib
import subprocess
import concurrent.futures
from pathlib import Path
from pydantic import model_validator
from lazyops.types.models import BaseModel
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, List, Any, Dict, Union
def add_to_env(envvar: str, envval: Any):
    """
    Helper for adding to the environment variable
    """
    envval = str(envval)
    if ' ' in envval:
        envval = f'"{envval}"'
    os.system(f"echo 'export {envvar}={envval}' >> ~/.bashrc")
    os.environ[envvar] = envval