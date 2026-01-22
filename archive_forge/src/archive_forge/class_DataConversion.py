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
class DataConversion:
    """
    DataConversion is a very simple container object which can be
    given an existing Dataset Element and provides methods to convert
    the Dataset into most other Element types.
    """

    def __init__(self, element):
        self._element = element

    def __call__(self, new_type, kdims=None, vdims=None, groupby=None, sort=False, **kwargs):
        """
        Generic conversion method for Dataset based Element
        types. Supply the Dataset Element type to convert to and
        optionally the key dimensions (kdims), value dimensions
        (vdims) and the dimensions.  to group over. Converted Columns
        can be automatically sorted via the sort option and kwargs can
        be passed through.
        """
        element_params = new_type.param.objects()
        kdim_param = element_params['kdims']
        vdim_param = element_params['vdims']
        if isinstance(kdim_param.bounds[1], int):
            ndim = min([kdim_param.bounds[1], len(kdim_param.default)])
        else:
            ndim = None
        nvdim = vdim_param.bounds[1] if isinstance(vdim_param.bounds[1], int) else None
        if kdims is None:
            kd_filter = groupby or []
            if not isinstance(kd_filter, list):
                kd_filter = [groupby]
            kdims = [kd for kd in self._element.kdims if kd not in kd_filter][:ndim]
        elif kdims and (not isinstance(kdims, list)):
            kdims = [kdims]
        if vdims is None:
            vdims = [d for d in self._element.vdims if d not in kdims][:nvdim]
        if vdims and (not isinstance(vdims, list)):
            vdims = [vdims]
        type_name = new_type.__name__
        for dim_type, dims in (('kdims', kdims), ('vdims', vdims)):
            min_d, max_d = element_params[dim_type].bounds
            if min_d is not None and len(dims) < min_d or (max_d is not None and len(dims) > max_d):
                raise ValueError(f'{type_name} {dim_type} must be between length {min_d} and {max_d}.')
        if groupby is None:
            groupby = [d for d in self._element.kdims if d not in kdims + vdims]
        elif groupby and (not isinstance(groupby, list)):
            groupby = [groupby]
        if self._element.interface.gridded:
            dropped_kdims = [kd for kd in self._element.kdims if kd not in groupby + kdims]
            if dropped_kdims:
                selected = self._element.reindex(groupby + kdims, vdims)
            else:
                selected = self._element
        elif issubclass(self._element.interface, PandasAPI):
            ds_dims = self._element.dimensions()
            ds_kdims = [self._element.get_dimension(d) if d in ds_dims else d for d in groupby + kdims]
            ds_vdims = [self._element.get_dimension(d) if d in ds_dims else d for d in vdims]
            selected = self._element.clone(kdims=ds_kdims, vdims=ds_vdims)
        else:
            selected = self._element.reindex(groupby + kdims, vdims)
        params = {'kdims': [selected.get_dimension(kd, strict=True) for kd in kdims], 'vdims': [selected.get_dimension(vd, strict=True) for vd in vdims], 'label': selected.label}
        if selected.group != selected.param.objects('existing')['group'].default:
            params['group'] = selected.group
        params.update(kwargs)
        if len(kdims) == selected.ndims or not groupby:
            params['dataset'] = self._element.dataset
            params['pipeline'] = self._element._pipeline
            element = new_type(selected, **params)
            return element.sort() if sort else element
        group = selected.groupby(groupby, container_type=HoloMap, group_type=new_type, **params)
        if sort:
            return group.map(lambda x: x.sort(), [new_type])
        else:
            return group