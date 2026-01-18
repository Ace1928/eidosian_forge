import importlib.resources as pkg_resources
import logging
from pathlib import Path
from typing import Any, List, Tuple
import yaml
from . import resources
from .deprecation_utils import deprecated
Returns the string of dictionary representation of the ReadMe.