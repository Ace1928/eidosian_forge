from __future__ import annotations as _annotations
import warnings
from typing import TYPE_CHECKING, Any
from typing_extensions import Literal, deprecated
from .._internal import _config
from ..warnings import PydanticDeprecatedSince20
class _ConfigMetaclass(type):

    def __getattr__(self, item: str) -> Any:
        try:
            obj = _config.config_defaults[item]
            warnings.warn(_config.DEPRECATION_MESSAGE, DeprecationWarning)
            return obj
        except KeyError as exc:
            raise AttributeError(f"type object '{self.__name__}' has no attribute {exc}") from exc