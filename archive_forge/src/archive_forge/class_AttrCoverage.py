import csv
import datetime
import os
from collections import OrderedDict, defaultdict
from typing import IO, Any, Dict, List, Optional, Set
from tabulate import tabulate
import onnx
from onnx import GraphProto, defs, helper
class AttrCoverage:

    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.values: Set[str] = set()

    def add(self, attr: onnx.AttributeProto) -> None:
        assert self.name in {None, attr.name}
        self.name = attr.name
        value = helper.get_attribute_value(attr)
        if isinstance(value, list):
            value = tuple(value)
        self.values.add(str(value))