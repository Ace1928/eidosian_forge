import dataclasses
import inspect
import io
import pathlib
from dataclasses import dataclass
from typing import List, Type, Dict, Iterator, Tuple, Set
import numpy as np
import pandas as pd
import cirq
from cirq._import import ModuleType
from cirq.protocols.json_serialization import ObjectFactory
@dataclass
class ModuleJsonTestSpec:
    name: str
    packages: List[ModuleType]
    test_data_path: pathlib.Path
    not_yet_serializable: List[str]
    should_not_be_serialized: List[str]
    resolver_cache: Dict[str, ObjectFactory]
    deprecated: Dict[str, str]
    custom_class_name_to_cirq_type: Dict[str, str] = dataclasses.field(default_factory=dict)
    tested_elsewhere: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self):
        return self.name

    def _get_all_public_classes(self) -> Iterator[Tuple[str, Type]]:
        for module in self.packages:
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) or inspect.ismodule(obj):
                    continue
                if name in self.should_not_be_serialized:
                    continue
                if not inspect.isclass(obj):
                    obj = obj.__class__
                if name.startswith('_'):
                    continue
                if inspect.isclass(obj) and inspect.isabstract(obj):
                    continue
                name = self.custom_class_name_to_cirq_type.get(name, name)
                yield (name, obj)

    def find_classes_that_should_serialize(self) -> Set[Tuple[str, Type]]:
        result: Set[Tuple[str, Type]] = set()
        result.update({(name, obj) for name, obj in self._get_all_public_classes()})
        result.update(self.get_resolver_cache_types())
        return result

    def get_resolver_cache_types(self) -> Set[Tuple[str, Type]]:
        result: Set[Tuple[str, Type]] = set()
        for k, v in self.resolver_cache.items():
            if isinstance(v, type):
                result.add((k, v))
        return result

    def get_all_names(self) -> Iterator[str]:

        def not_module_or_function(x):
            return not (inspect.ismodule(x) or inspect.isfunction(x))
        for m in self.packages:
            for name, _ in inspect.getmembers(m, not_module_or_function):
                yield name
        for name, _ in self.get_resolver_cache_types():
            yield name

    def all_test_data_keys(self) -> List[str]:
        seen = set()
        for file in self.test_data_path.iterdir():
            name = str(file.absolute())
            if name.endswith('.json') or name.endswith('.repr'):
                seen.add(name[:-len('.json')])
            elif name.endswith('.json_inward') or name.endswith('.repr_inward'):
                seen.add(name[:-len('.json_inward')])
        return sorted(seen)