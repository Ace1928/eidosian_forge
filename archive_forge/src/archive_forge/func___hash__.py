import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def __hash__(self):
    return hash(self.to_json_string(ignore_metadata=True))