import json
from typing import IO, List, Tuple, Any
from .parser import Parser
from .symbols import (
def write_long(self, value):
    self._parser.advance(Long())
    self.write_value(value)