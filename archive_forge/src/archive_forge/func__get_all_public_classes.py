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