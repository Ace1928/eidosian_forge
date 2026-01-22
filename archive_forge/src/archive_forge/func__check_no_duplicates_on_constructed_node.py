import re
import textwrap
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import yaml
from huggingface_hub import DatasetCardData
from ..config import METADATA_CONFIGS_FIELD
from ..info import DatasetInfo, DatasetInfosDict
from ..naming import _split_re
from ..utils.logging import get_logger
from .deprecation_utils import deprecated
def _check_no_duplicates_on_constructed_node(self, node):
    keys = [self.constructed_objects[key_node] for key_node, _ in node.value]
    keys = [tuple(key) if isinstance(key, list) else key for key in keys]
    counter = Counter(keys)
    duplicate_keys = [key for key in counter if counter[key] > 1]
    if duplicate_keys:
        raise TypeError(f'Got duplicate yaml keys: {duplicate_keys}')