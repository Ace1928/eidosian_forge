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
def to_readme(self, path: Path):
    if path.exists():
        with open(path, encoding='utf-8') as readme_file:
            readme_content = readme_file.read()
    else:
        readme_content = None
    updated_readme_content = self._to_readme(readme_content)
    with open(path, 'w', encoding='utf-8') as readme_file:
        readme_file.write(updated_readme_content)