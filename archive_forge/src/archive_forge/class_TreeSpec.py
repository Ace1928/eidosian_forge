import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
@dataclasses.dataclass
class TreeSpec:
    type: Any
    context: Context
    children_specs: List['TreeSpec']

    def __post_init__(self) -> None:
        self.num_leaves: int = sum([spec.num_leaves for spec in self.children_specs])

    def __repr__(self, indent: int=0) -> str:
        repr_prefix: str = f'TreeSpec({self.type.__name__}, {self.context}, ['
        children_specs_str: str = ''
        if len(self.children_specs):
            indent += 2
            children_specs_str += self.children_specs[0].__repr__(indent)
            children_specs_str += ',' if len(self.children_specs) > 1 else ''
            children_specs_str += ','.join(['\n' + ' ' * indent + child.__repr__(indent) for child in self.children_specs[1:]])
        repr_suffix: str = f'{children_specs_str}])'
        return repr_prefix + repr_suffix

    def is_leaf(self) -> bool:
        return isinstance(self, LeafSpec)