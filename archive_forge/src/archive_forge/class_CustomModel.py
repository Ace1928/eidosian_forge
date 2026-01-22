from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
class CustomModel:
    """ Represent a custom (user-defined) Bokeh model.

    """

    def __init__(self, cls: type[HasProps]) -> None:
        self.cls = cls

    @property
    def name(self) -> str:
        return self.cls.__name__

    @property
    def full_name(self) -> str:
        name = self.cls.__module__ + '.' + self.name
        return name.replace('__main__.', '')

    @property
    def file(self) -> str | None:
        module = sys.modules[self.cls.__module__]
        if hasattr(module, '__file__') and (file := module.__file__) is not None:
            return abspath(file)
        else:
            return None

    @property
    def path(self) -> str:
        path = getattr(self.cls, '__base_path__', None)
        if path is not None:
            return path
        elif self.file is not None:
            return dirname(self.file)
        else:
            return os.getcwd()

    @property
    def implementation(self) -> Implementation:
        impl = getattr(self.cls, '__implementation__')
        if isinstance(impl, str):
            if '\n' not in impl and impl.endswith(exts):
                impl = FromFile(impl if isabs(impl) else join(self.path, impl))
            else:
                impl = TypeScript(impl)
        if isinstance(impl, Inline) and impl.file is None:
            file = f'{(self.file + ':' if self.file else '')}{self.name}.ts'
            impl = impl.__class__(impl.code, file)
        return impl

    @property
    def dependencies(self) -> dict[str, str]:
        return getattr(self.cls, '__dependencies__', {})

    @property
    def module(self) -> str:
        return f'custom/{snakify(self.full_name)}'