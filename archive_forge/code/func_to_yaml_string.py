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
def to_yaml_string(self) -> str:
    return yaml.safe_dump({key.replace('_', '-') if key in self._FIELDS_WITH_DASHES else key: value for key, value in self.items()}, sort_keys=False, allow_unicode=True, encoding='utf-8').decode('utf-8')