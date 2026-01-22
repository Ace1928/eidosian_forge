import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
class SplitDict(dict):
    """Split info object."""

    def __init__(self, *args, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def __getitem__(self, key: Union[SplitBase, str]):
        if str(key) in self:
            return super().__getitem__(str(key))
        else:
            instructions = make_file_instructions(name=self.dataset_name, split_infos=self.values(), instruction=key)
            return SubSplitInfo(instructions)

    def __setitem__(self, key: Union[SplitBase, str], value: SplitInfo):
        if key != value.name:
            raise ValueError(f"Cannot add elem. (key mismatch: '{key}' != '{value.name}')")
        super().__setitem__(key, value)

    def add(self, split_info: SplitInfo):
        """Add the split info."""
        if split_info.name in self:
            raise ValueError(f'Split {split_info.name} already present')
        split_info.dataset_name = self.dataset_name
        super().__setitem__(split_info.name, split_info)

    @property
    def total_num_examples(self):
        """Return the total number of examples."""
        return sum((s.num_examples for s in self.values()))

    @classmethod
    def from_split_dict(cls, split_infos: Union[List, Dict], dataset_name: Optional[str]=None):
        """Returns a new SplitDict initialized from a Dict or List of `split_infos`."""
        if isinstance(split_infos, dict):
            split_infos = list(split_infos.values())
        if dataset_name is None:
            dataset_name = split_infos[0].get('dataset_name') if split_infos else None
        split_dict = cls(dataset_name=dataset_name)
        for split_info in split_infos:
            if isinstance(split_info, dict):
                split_info = SplitInfo(**split_info)
            split_dict.add(split_info)
        return split_dict

    def to_split_dict(self):
        """Returns a list of SplitInfo protos that we have."""
        out = []
        for split_name, split_info in self.items():
            split_info = copy.deepcopy(split_info)
            split_info.name = split_name
            out.append(split_info)
        return out

    def copy(self):
        return SplitDict.from_split_dict(self.to_split_dict(), self.dataset_name)

    def _to_yaml_list(self) -> list:
        out = [asdict(s) for s in self.to_split_dict()]
        for split_info_dict in out:
            split_info_dict.pop('shard_lengths', None)
        for split_info_dict in out:
            split_info_dict.pop('dataset_name', None)
        return out

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> 'SplitDict':
        return cls.from_split_dict(yaml_data)