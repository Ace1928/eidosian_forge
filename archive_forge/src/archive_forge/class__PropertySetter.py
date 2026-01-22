import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
class _PropertySetter:

    def __init__(self, prop: str, schema: dict) -> None:
        self.prop = prop
        self.schema = schema

    def __get__(self, obj, cls):
        self.obj = obj
        self.cls = cls
        self.__doc__ = self.schema['description'].replace('__', '**')
        property_name = f'{self.prop}'[0].upper() + f'{self.prop}'[1:]
        if hasattr(vegalite, property_name):
            altair_prop = getattr(vegalite, property_name)
            parameter_index = altair_prop.__doc__.find('Parameters\n')
            if parameter_index > -1:
                self.__doc__ = altair_prop.__doc__[:parameter_index].replace('    ', '') + self.__doc__ + textwrap.dedent(f'\n\n    {altair_prop.__doc__[parameter_index:]}')
            else:
                self.__doc__ = altair_prop.__doc__.replace('    ', '') + '\n' + self.__doc__
            self.__signature__ = inspect.signature(altair_prop)
            self.__wrapped__ = inspect.getfullargspec(altair_prop)
            self.__name__ = altair_prop.__name__
        else:
            pass
        return self

    def __call__(self, *args, **kwargs):
        obj = self.obj.copy()
        obj[self.prop] = args[0] if args else kwargs
        return obj