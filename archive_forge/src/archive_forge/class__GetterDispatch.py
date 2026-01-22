import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union
from .helpers import make_path
class _GetterDispatch:

    def __init__(self, initialdata, default_getter: Callable):
        self.default_getter = default_getter
        self.data = initialdata

    def get_fn_for_type(self, type_):
        return self.data.get(type_, self.default_getter)

    def get_typed_value(self, type_, name):
        get_fn = self.get_fn_for_type(type_)
        return get_fn(name)