import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def _spec_to_obj(self_, spec, dynamic=True, intermediate=True):
    """
        Resolves a dependency specification into lists of explicit
        parameter dependencies and dynamic dependencies.

        Dynamic dependencies are specifications to be resolved when
        the sub-object whose parameters are being depended on is
        defined.

        During class creation dynamic=False which means sub-object
        dependencies are not resolved. At instance creation and
        whenever a sub-object is set on an object this method will be
        invoked to determine whether the dependency is available.

        For sub-object dependencies we also return dependencies for
        every part of the path, e.g. for a dependency specification
        like "a.b.c" we return dependencies for sub-object "a" and the
        sub-sub-object "b" in addition to the dependency on the actual
        parameter "c" on object "b". This is to ensure that if a
        sub-object is swapped out we are notified and can update the
        dynamic dependency to the new object. Even if a sub-object
        dependency can only partially resolved, e.g. if object "a"
        does not yet have a sub-object "b" we must watch for changes
        to "b" on sub-object "a" in case such a subobject is put in "b".
        """
    if isinstance(spec, Parameter):
        inst = spec.owner if isinstance(spec.owner, Parameterized) else None
        cls = spec.owner if inst is None else type(inst)
        info = PInfo(inst=inst, cls=cls, name=spec.name, pobj=spec, what='value')
        return ([] if intermediate == 'only' else [info], [])
    obj, attr, what = _parse_dependency_spec(spec)
    if obj is None:
        src = self_.self_or_cls
    elif not dynamic:
        return ([], [DInfo(spec=spec)])
    else:
        if not hasattr(self_.self_or_cls, obj.split('.')[1]):
            raise AttributeError(f'Dependency {obj[1:]!r} could not be resolved, {self_.self_or_cls} has no parameter or attribute {obj.split('.')[1]!r}. Ensure the object being depended on is declared before calling the Parameterized constructor.')
        src = _getattrr(self_.self_or_cls, obj[1:], None)
        if src is None:
            path = obj[1:].split('.')
            deps = []
            if len(path) >= 1 and intermediate:
                sub_src = None
                subpath = path
                while sub_src is None and subpath:
                    subpath = subpath[:-1]
                    sub_src = _getattrr(self_.self_or_cls, '.'.join(subpath), None)
                if subpath:
                    subdeps, _ = self_._spec_to_obj('.'.join(path[:len(subpath) + 1]), dynamic, intermediate)
                    deps += subdeps
            return (deps, [] if intermediate == 'only' else [DInfo(spec=spec)])
    cls, inst = (src, None) if isinstance(src, type) else (type(src), src)
    if attr == 'param':
        deps, dynamic_deps = self_._spec_to_obj(obj[1:], dynamic, intermediate)
        for p in src.param:
            param_deps, param_dynamic_deps = src.param._spec_to_obj(p, dynamic, intermediate)
            deps += param_deps
            dynamic_deps += param_dynamic_deps
        return (deps, dynamic_deps)
    elif attr in src.param:
        info = PInfo(inst=inst, cls=cls, name=attr, pobj=src.param[attr], what=what)
    elif hasattr(src, attr):
        attr_obj = getattr(src, attr)
        if isinstance(attr_obj, Parameterized):
            return ([], [])
        elif isinstance(attr_obj, (FunctionType, MethodType)):
            info = MInfo(inst=inst, cls=cls, name=attr, method=attr_obj)
        else:
            raise AttributeError(f'Attribute {attr!r} could not be resolved on {src}.')
    elif getattr(src, 'abstract', None):
        return ([], [] if intermediate == 'only' else [DInfo(spec=spec)])
    else:
        raise AttributeError(f'Attribute {attr!r} could not be resolved on {src}.')
    if obj is None or not intermediate:
        return ([info], [])
    deps, dynamic_deps = self_._spec_to_obj(obj[1:], dynamic, intermediate)
    if intermediate != 'only':
        deps.append(info)
    return (deps, dynamic_deps)