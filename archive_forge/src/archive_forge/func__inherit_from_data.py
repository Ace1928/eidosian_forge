from __future__ import annotations
from typing import (
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.indexes.base import Index
def _inherit_from_data(name: str, delegate: type, cache: bool=False, wrap: bool=False):
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
    attr = getattr(delegate, name)
    if isinstance(attr, property) or type(attr).__name__ == 'getset_descriptor':
        if cache:

            def cached(self):
                return getattr(self._data, name)
            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            method = cache_readonly(cached)
        else:

            def fget(self):
                result = getattr(self._data, name)
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name)
                return result

            def fset(self, value) -> None:
                setattr(self._data, name, value)
            fget.__name__ = name
            fget.__doc__ = attr.__doc__
            method = property(fget, fset)
    elif not callable(attr):
        method = attr
    else:

        def method(self, *args, **kwargs):
            if 'inplace' in kwargs:
                raise ValueError(f'cannot use inplace with {type(self).__name__}')
            result = attr(self._data, *args, **kwargs)
            if wrap:
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                return Index(result, name=self.name)
            return result
        method.__name__ = name
        method.__doc__ = attr.__doc__
    return method