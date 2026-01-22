from collections.abc import Iterable
import logging
from pyomo.common.collections import ComponentSet
from pyomo.common.config import (
from pyomo.common.errors import ApplicationError, PyomoException
from pyomo.core.base import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import ObjectiveType, setup_pyros_logger
from pyomo.contrib.pyros.uncertainty_sets import UncertaintySet
class InputDataStandardizer(object):
    """
    Standardizer for objects castable to a list of Pyomo
    component types.

    Parameters
    ----------
    ctype : type
        Pyomo component type, such as Component, Var or Param.
    cdatatype : type
        Corresponding Pyomo component data type, such as
        _ComponentData, _VarData, or _ParamData.
    ctype_validator : callable, optional
        Validator function for objects of type `ctype`.
    cdatatype_validator : callable, optional
        Validator function for objects of type `cdatatype`.
    allow_repeats : bool, optional
        True to allow duplicate component data entries in final
        list to which argument is cast, False otherwise.

    Attributes
    ----------
    ctype
    cdatatype
    ctype_validator
    cdatatype_validator
    allow_repeats
    """

    def __init__(self, ctype, cdatatype, ctype_validator=None, cdatatype_validator=None, allow_repeats=False):
        """Initialize self (see class docstring)."""
        self.ctype = ctype
        self.cdatatype = cdatatype
        self.ctype_validator = ctype_validator
        self.cdatatype_validator = cdatatype_validator
        self.allow_repeats = allow_repeats

    def standardize_ctype_obj(self, obj):
        """
        Standardize object of type ``self.ctype`` to list
        of objects of type ``self.cdatatype``.
        """
        if self.ctype_validator is not None:
            self.ctype_validator(obj)
        return list(obj.values())

    def standardize_cdatatype_obj(self, obj):
        """
        Standardize object of type ``self.cdatatype`` to
        ``[obj]``.
        """
        if self.cdatatype_validator is not None:
            self.cdatatype_validator(obj)
        return [obj]

    def __call__(self, obj, from_iterable=None, allow_repeats=None):
        """
        Cast object to a flat list of Pyomo component data type
        entries.

        Parameters
        ----------
        obj : object
            Object to be cast.
        from_iterable : Iterable or None, optional
            Iterable from which `obj` obtained, if any.
        allow_repeats : bool or None, optional
            True if list can contain repeated entries,
            False otherwise.

        Raises
        ------
        TypeError
            If all entries in the resulting list
            are not of type ``self.cdatatype``.
        ValueError
            If the resulting list contains duplicate entries.
        """
        if allow_repeats is None:
            allow_repeats = self.allow_repeats
        if isinstance(obj, self.ctype):
            ans = self.standardize_ctype_obj(obj)
        elif isinstance(obj, self.cdatatype):
            ans = self.standardize_cdatatype_obj(obj)
        elif isinstance(obj, Iterable) and (not isinstance(obj, str)):
            ans = []
            for item in obj:
                ans.extend(self.__call__(item, from_iterable=obj))
        else:
            from_iterable_qual = f' (entry of iterable {from_iterable})' if from_iterable is not None else ''
            raise TypeError(f'Input object {obj!r}{from_iterable_qual} is not of valid component type {self.ctype.__name__} or component data type {self.cdatatype.__name__}.')
        if not allow_repeats and len(ans) != len(ComponentSet(ans)):
            comp_name_list = [comp.name for comp in ans]
            raise ValueError(f'Standardized component list {comp_name_list} derived from input {obj} contains duplicate entries.')
        return ans

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        return f'{self.cdatatype.__name__}, {self.ctype.__name__}, or Iterable of {self.cdatatype.__name__}/{self.ctype.__name__}'