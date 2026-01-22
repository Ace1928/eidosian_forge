import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def __set_name__(self, owner: Any, name: str) -> None:
    if self.path_or_name is None:
        self.path_or_name = name
    super().__set_name__(owner, name)