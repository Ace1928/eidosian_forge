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
@staticmethod
def _raise_if_data_files_field_not_valid(metadata_config: dict):
    yaml_data_files = metadata_config.get('data_files')
    if yaml_data_files is not None:
        yaml_error_message = textwrap.dedent(f"\n                Expected data_files in YAML to be either a string or a list of strings\n                or a list of dicts with two keys: 'split' and 'path', but got {yaml_data_files}\n                Examples of data_files in YAML:\n\n                   data_files: data.csv\n\n                   data_files: data/*.png\n\n                   data_files:\n                    - part0/*\n                    - part1/*\n\n                   data_files:\n                    - split: train\n                      path: train/*\n                    - split: test\n                      path: test/*\n\n                   data_files:\n                    - split: train\n                      path:\n                      - train/part1/*\n                      - train/part2/*\n                    - split: test\n                      path: test/*\n\n                PS: some symbols like dashes '-' are not allowed in split names\n                ")
        if not isinstance(yaml_data_files, (list, str)):
            raise ValueError(yaml_error_message)
        if isinstance(yaml_data_files, list):
            for yaml_data_files_item in yaml_data_files:
                if not isinstance(yaml_data_files_item, (str, dict)) or (isinstance(yaml_data_files_item, dict) and (not (len(yaml_data_files_item) == 2 and 'split' in yaml_data_files_item and re.match(_split_re, yaml_data_files_item['split']) and isinstance(yaml_data_files_item.get('path'), (str, list))))):
                    raise ValueError(yaml_error_message)