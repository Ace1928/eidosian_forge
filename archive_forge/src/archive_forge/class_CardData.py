import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
@dataclass
class CardData:
    """Structure containing metadata from a RepoCard.

    [`CardData`] is the parent class of [`ModelCardData`] and [`DatasetCardData`].

    Metadata can be exported as a dictionary or YAML. Export can be customized to alter the representation of the data
    (example: flatten evaluation results). `CardData` behaves as a dictionary (can get, pop, set values) but do not
    inherit from `dict` to allow this export step.
    """

    def __init__(self, ignore_metadata_errors: bool=False, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Converts CardData to a dict.

        Returns:
            `dict`: CardData represented as a dictionary ready to be dumped to a YAML
            block for inclusion in a README.md file.
        """
        data_dict = copy.deepcopy(self.__dict__)
        self._to_dict(data_dict)
        return _remove_none(data_dict)

    def _to_dict(self, data_dict):
        """Use this method in child classes to alter the dict representation of the data. Alter the dict in-place.

        Args:
            data_dict (`dict`): The raw dict representation of the card data.
        """
        pass

    def to_yaml(self, line_break=None) -> str:
        """Dumps CardData to a YAML block for inclusion in a README.md file.

        Args:
            line_break (str, *optional*):
                The line break to use when dumping to yaml.

        Returns:
            `str`: CardData represented as a YAML block.
        """
        return yaml_dump(self.to_dict(), sort_keys=False, line_break=line_break).strip()

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return self.to_yaml()

    def get(self, key: str, default: Any=None) -> Any:
        """Get value for a given metadata key."""
        return self.__dict__.get(key, default)

    def pop(self, key: str, default: Any=None) -> Any:
        """Pop value for a given metadata key."""
        return self.__dict__.pop(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get value for a given metadata key."""
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value for a given metadata key."""
        self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a given metadata key is set."""
        return key in self.__dict__

    def __len__(self) -> int:
        """Return the number of metadata keys set."""
        return len(self.__dict__)