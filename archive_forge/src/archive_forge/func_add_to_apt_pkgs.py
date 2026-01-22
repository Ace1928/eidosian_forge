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
def add_to_apt_pkgs(*pkg: str):
    """
    Helper for adding to the apt packages
    """
    global APT_PKGS
    APT_PKGS.extend(pkg)