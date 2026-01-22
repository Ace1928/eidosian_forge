import copy
import json
import os
import re
from typing import Any, Dict, List, Tuple, Union
from huggingface_hub import HfFolder
from packaging import version
from transformers import PretrainedConfig
from transformers import __version__ as transformers_version_str
from .utils import logging
from .version import __version__
@classmethod
def _re_configuration_file(cls):
    return re.compile(f'{cls.FULL_CONFIGURATION_FILE.split('.')[0]}(.*)\\.json')