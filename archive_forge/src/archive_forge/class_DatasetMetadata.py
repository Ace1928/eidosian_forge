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
@deprecated('Use `huggingface_hub.DatasetCardData` instead.')
class DatasetMetadata(dict):
    _FIELDS_WITH_DASHES = {'train_eval_index'}

    @classmethod
    def from_readme(cls, path: Union[Path, str]) -> 'DatasetMetadata':
        """Loads and validates the dataset metadata from its dataset card (README.md)

        Args:
            path (:obj:`Path`): Path to the dataset card (its README.md file)

        Returns:
            :class:`DatasetMetadata`: The dataset's metadata

        Raises:
            :obj:`TypeError`: If the dataset's metadata is invalid
        """
        with open(path, encoding='utf-8') as readme_file:
            yaml_string, _ = _split_yaml_from_readme(readme_file.read())
        if yaml_string is not None:
            return cls.from_yaml_string(yaml_string)
        else:
            return cls()

    def to_readme(self, path: Path):
        if path.exists():
            with open(path, encoding='utf-8') as readme_file:
                readme_content = readme_file.read()
        else:
            readme_content = None
        updated_readme_content = self._to_readme(readme_content)
        with open(path, 'w', encoding='utf-8') as readme_file:
            readme_file.write(updated_readme_content)

    def _to_readme(self, readme_content: Optional[str]=None) -> str:
        if readme_content is not None:
            _, content = _split_yaml_from_readme(readme_content)
            full_content = '---\n' + self.to_yaml_string() + '---\n' + content
        else:
            full_content = '---\n' + self.to_yaml_string() + '---\n'
        return full_content

    @classmethod
    def from_yaml_string(cls, string: str) -> 'DatasetMetadata':
        """Loads and validates the dataset metadata from a YAML string

        Args:
            string (:obj:`str`): The YAML string

        Returns:
            :class:`DatasetMetadata`: The dataset's metadata

        Raises:
            :obj:`TypeError`: If the dataset's metadata is invalid
        """
        metadata_dict = yaml.load(string, Loader=_NoDuplicateSafeLoader) or {}
        metadata_dict = {key.replace('-', '_') if key.replace('-', '_') in cls._FIELDS_WITH_DASHES else key: value for key, value in metadata_dict.items()}
        return cls(**metadata_dict)

    def to_yaml_string(self) -> str:
        return yaml.safe_dump({key.replace('_', '-') if key in self._FIELDS_WITH_DASHES else key: value for key, value in self.items()}, sort_keys=False, allow_unicode=True, encoding='utf-8').decode('utf-8')