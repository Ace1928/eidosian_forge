from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload
class Meta(type):

    def __setattr__(self, key: str, value: object) -> None:
        obj = self.__dict__.get(key, None)
        if type(obj) is classproperty:
            return obj.__set__(self, value)
        return super().__setattr__(key, value)