import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
class SubclassOnlyABC:

    def __new__(cls, *args: Any, **kwargs: Any) -> T:
        if SubclassOnlyABC in cls.__bases__:
            raise TypeError(f'Abstract class {cls.__name__} cannot be instantiated')
        return super().__new__(cls)