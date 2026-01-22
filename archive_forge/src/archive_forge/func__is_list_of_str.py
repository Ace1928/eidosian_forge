import importlib.util
import os
from collections import namedtuple
from typing import Any, List, Optional
from pip._vendor import tomli
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._internal.exceptions import (
def _is_list_of_str(obj: Any) -> bool:
    return isinstance(obj, list) and all((isinstance(item, str) for item in obj))