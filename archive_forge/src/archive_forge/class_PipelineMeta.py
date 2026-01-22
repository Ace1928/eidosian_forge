import copy
import types
from contextlib import contextmanager
from functools import wraps
import numpy as np
import pandas as pd  # noqa
import param
from param.parameterized import ParameterizedMetaclass
from .. import util as core_util
from ..accessors import Redim
from ..dimension import (
from ..element import Element
from ..ndmapping import MultiDimensionalMapping
from ..spaces import DynamicMap, HoloMap
from .array import ArrayInterface
from .cudf import cuDFInterface  # noqa (API import)
from .dask import DaskInterface  # noqa (API import)
from .dictionary import DictInterface  # noqa (API import)
from .grid import GridInterface  # noqa (API import)
from .ibis import IbisInterface  # noqa (API import)
from .image import ImageInterface  # noqa (API import)
from .interface import Interface, iloc, ndloc
from .multipath import MultiInterface  # noqa (API import)
from .pandas import PandasAPI, PandasInterface  # noqa (API import)
from .spatialpandas import SpatialPandasInterface  # noqa (API import)
from .spatialpandas_dask import DaskSpatialPandasInterface  # noqa (API import)
from .xarray import XArrayInterface  # noqa (API import)
class PipelineMeta(ParameterizedMetaclass):
    blacklist = ['__init__', 'clone']
    disable = False

    def __new__(mcs, classname, bases, classdict):
        for method_name in classdict:
            method_fn = classdict[method_name]
            if method_name in mcs.blacklist or method_name.startswith('_'):
                continue
            elif isinstance(method_fn, types.FunctionType):
                classdict[method_name] = mcs.pipelined(method_fn, method_name)
        inst = type.__new__(mcs, classname, bases, classdict)
        return inst

    @staticmethod
    def pipelined(method_fn, method_name):

        @wraps(method_fn)
        def pipelined_fn(*args, **kwargs):
            from ...operation.element import method as method_op
            inst = args[0]
            inst_pipeline = copy.copy(getattr(inst, '_pipeline', None))
            in_method = inst._in_method
            if not in_method:
                inst._in_method = True
            try:
                result = method_fn(*args, **kwargs)
                if PipelineMeta.disable:
                    return result
                op = method_op.instance(input_type=type(inst), method_name=method_name, args=list(args[1:]), kwargs=kwargs)
                if not in_method:
                    if isinstance(result, Dataset):
                        result._pipeline = inst_pipeline.instance(operations=inst_pipeline.operations + [op], output_type=type(result))
                    elif isinstance(result, MultiDimensionalMapping):
                        for key, element in result.items():
                            if isinstance(element, Dataset):
                                getitem_op = method_op.instance(input_type=type(result), method_name='__getitem__', args=[key])
                                element._pipeline = inst_pipeline.instance(operations=inst_pipeline.operations + [op, getitem_op], output_type=type(result))
            finally:
                if not in_method:
                    inst._in_method = False
            return result
        return pipelined_fn