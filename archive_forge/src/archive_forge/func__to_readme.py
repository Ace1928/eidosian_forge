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
def _to_readme(self, readme_content: Optional[str]=None) -> str:
    if readme_content is not None:
        _, content = _split_yaml_from_readme(readme_content)
        full_content = '---\n' + self.to_yaml_string() + '---\n' + content
    else:
        full_content = '---\n' + self.to_yaml_string() + '---\n'
    return full_content