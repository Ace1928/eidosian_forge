import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
class Converter(object):
    """
    Conversion between rpy2's low-level and high-level proxy objects
    for R objects, and Python (no R) objects.

    Converter objects can be added, the result being
    a Converter objects combining the translation rules from the
    different converters.
    """
    _name: str
    _rpy2py_nc_map: typing.Dict[typing.Type[rpy2.rinterface_lib.sexp.Sexp], NameClassMap]
    _lineage: typing.Tuple[str, ...]
    name = property(lambda self: self._name)
    py2rpy = property(lambda self: self._py2rpy)
    rpy2py = property(lambda self: self._rpy2py)
    rpy2py_nc_map = property(lambda self: self._rpy2py_nc_map)
    rpy2py_nc_name = rpy2py_nc_map
    lineage = property(lambda self: self._lineage)

    def __init__(self, name: str, template: typing.Optional['Converter']=None):
        py2rpy, rpy2py = Converter.make_dispatch_functions()
        self._name = name
        self._py2rpy = py2rpy
        self._rpy2py = rpy2py
        self._rpy2py_nc_map = {}
        lineage: typing.Tuple[str, ...]
        if template is None:
            lineage = tuple()
        else:
            _ = list(template.lineage)
            _.append(name)
            lineage = tuple(_)
            overlay_converter(template, self)
        self._lineage = lineage

    def __add__(self, converter: 'Converter') -> 'Converter':
        assert isinstance(converter, Converter)
        new_name = '%s + %s' % (self.name, converter.name)
        result_converter = Converter(new_name, template=self)
        overlay_converter(converter, result_converter)
        return result_converter

    def context(self) -> 'ConversionContext':
        """
        Create a Conversion context to use in a `with` statement.

        >>> with conversion_rules.context() as cv:
        ...     # Do something while using those conversion_rules.
        >>> # Do something else whith the earlier conversion rules restored.

        The conversion context is a *copy* of the converter object.

        :return: A :class:`ConversionContext`
        """
        return ConversionContext(self)

    def __enter__(self):
        raise Exception("Use the converter's method context instead.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __str__(self):
        res = [str(type(self))]
        for subcv in ('py2rpy', 'rpy2py'):
            res.append(subcv)
            for cls in getattr(self, subcv).registry.keys():
                res.append(f'- {cls.__module__}.{cls.__name__}')
                if subcv == 'rpy2py':
                    ncmap = self._rpy2py_nc_map.get(cls)
                    if ncmap:
                        for k, v in ncmap._map.items():
                            res.append(f'  - {k} (in {v.__module__})')
            res.append('---')
        return os.linesep.join(res)

    @staticmethod
    def make_dispatch_functions():
        py2rpy = singledispatch(_py2rpy)
        rpy2py = singledispatch(_rpy2py)
        return (py2rpy, rpy2py)

    def rclass_map_context(self, cls, d: typing.Dict[str, typing.Type]):
        return NameClassMapContext(self.rpy2py_nc_map[cls], d)