import contextlib
import functools
import inspect
import math
import warnings
class _deprecated_property(type(obj)):

    def __get__(self, instance, owner=None):
        if instance is not None or (owner is not None and isinstance(self, classproperty)):
            emit_warning()
        return super().__get__(instance, owner)

    def __set__(self, instance, value):
        if instance is not None:
            emit_warning()
        return super().__set__(instance, value)

    def __delete__(self, instance):
        if instance is not None:
            emit_warning()
        return super().__delete__(instance)

    def __set_name__(self, owner, set_name):
        nonlocal name
        if name == '<lambda>':
            name = set_name