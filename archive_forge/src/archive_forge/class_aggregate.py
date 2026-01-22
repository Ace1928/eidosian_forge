import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
class aggregate(LineAggregationOperation):
    """
    aggregate implements 2D binning for any valid HoloViews Element
    type using datashader. I.e., this operation turns a HoloViews
    Element or overlay of Elements into an Image or an overlay of
    Images by rasterizing it. This allows quickly aggregating large
    datasets computing a fixed-sized representation independent
    of the original dataset size.

    By default it will simply count the number of values in each bin
    but other aggregators can be supplied implementing mean, max, min
    and other reduction operations.

    The bins of the aggregate are defined by the width and height and
    the x_range and y_range. If x_sampling or y_sampling are supplied
    the operation will ensure that a bin is no smaller than the minimum
    sampling distance by reducing the width and height when zoomed in
    beyond the minimum sampling distance.

    By default, the PlotSize stream is applied when this operation
    is used dynamically, which means that the height and width
    will automatically be set to match the inner dimensions of
    the linked plot.
    """

    @classmethod
    def get_agg_data(cls, obj, category=None):
        """
        Reduces any Overlay or NdOverlay of Elements into a single
        xarray Dataset that can be aggregated.
        """
        paths = []
        if isinstance(obj, Graph):
            obj = obj.edgepaths
        kdims = list(obj.kdims)
        vdims = list(obj.vdims)
        dims = obj.dimensions()[:2]
        if isinstance(obj, Path):
            glyph = 'line'
            for p in obj.split(datatype='dataframe'):
                paths.append(p)
        elif isinstance(obj, CompositeOverlay):
            element = None
            for key, el in obj.data.items():
                x, y, element, glyph = cls.get_agg_data(el)
                dims = (x, y)
                df = PandasInterface.as_dframe(element)
                if isinstance(obj, NdOverlay):
                    df = df.assign(**dict(zip(obj.dimensions('key', True), key)))
                paths.append(df)
            if element is None:
                dims = None
            else:
                kdims += element.kdims
                vdims = element.vdims
        elif isinstance(obj, Element):
            glyph = 'line' if isinstance(obj, Curve) else 'points'
            paths.append(PandasInterface.as_dframe(obj))
        if dims is None or len(dims) != 2:
            return (None, None, None, None)
        else:
            x, y = dims
        if len(paths) > 1:
            if glyph == 'line':
                path = paths[0][:1]
                if isinstance(path, dd.DataFrame):
                    path = path.compute()
                empty = path.copy()
                empty.iloc[0, :] = (np.nan,) * empty.shape[1]
                paths = [elem for p in paths for elem in (p, empty)][:-1]
            if all((isinstance(path, dd.DataFrame) for path in paths)):
                df = dd.concat(paths)
            else:
                paths = [p.compute() if isinstance(p, dd.DataFrame) else p for p in paths]
                df = pd.concat(paths)
        else:
            df = paths[0] if paths else pd.DataFrame([], columns=[x.name, y.name])
        if category and df[category].dtype.name != 'category':
            df[category] = df[category].astype('category')
        is_custom = isinstance(df, dd.DataFrame) or cuDFInterface.applies(df)
        if any((not is_custom and len(df[d.name]) and isinstance(df[d.name].values[0], cftime_types) or df[d.name].dtype.kind in ['M', 'u'] for d in (x, y))):
            df = df.copy()
        for d in (x, y):
            vals = df[d.name]
            if not is_custom and len(vals) and isinstance(vals.values[0], cftime_types):
                vals = cftime_to_timestamp(vals, 'ns')
            elif vals.dtype.kind == 'M':
                vals = vals.astype('datetime64[ns]')
            elif vals.dtype == np.uint64:
                raise TypeError(f'Dtype of uint64 for column {d.name} is not supported.')
            elif vals.dtype.kind == 'u':
                pass
            else:
                continue
            df[d.name] = cast_array_to_int64(vals)
        return (x, y, Dataset(df, kdims=kdims, vdims=vdims), glyph)

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        sel_fn = getattr(self.p, 'selector', None)
        if hasattr(agg_fn, 'cat_column'):
            category = agg_fn.cat_column
        else:
            category = agg_fn.column if isinstance(agg_fn, ds.count_cat) else None
        if overlay_aggregate.applies(element, agg_fn, line_width=self.p.line_width):
            params = dict({p: v for p, v in self.param.values().items() if p != 'name'}, dynamic=False, **{p: v for p, v in self.p.items() if p not in ('name', 'dynamic')})
            return overlay_aggregate(element, **params)
        if element._plot_id in self._precomputed:
            x, y, data, glyph = self._precomputed[element._plot_id]
        else:
            x, y, data, glyph = self.get_agg_data(element, category)
        if self.p.precompute:
            self._precomputed[element._plot_id] = (x, y, data, glyph)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = self._get_sampling(element, x, y)
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        params = self._get_agg_params(element, x, y, agg_fn, (x0, y0, x1, y1))
        if x is None or y is None or width == 0 or (height == 0):
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)
        elif getattr(data, 'interface', None) is not DaskInterface and (not len(data)):
            empty_val = 0 if isinstance(agg_fn, ds.count) else np.nan
            xarray = xr.DataArray(np.full((height, width), empty_val), dims=[y.name, x.name], coords={x.name: xs, y.name: ys})
            return self.p.element_type(xarray, **params)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        agg_kwargs = {}
        if self.p.line_width and glyph == 'line' and (ds_version >= Version('0.14.0')):
            agg_kwargs['line_width'] = self.p.line_width
        dfdata = PandasInterface.as_dframe(data)
        cvs_fn = getattr(cvs, glyph)
        if sel_fn:
            if isinstance(params['vdims'], (Dimension, str)):
                params['vdims'] = [params['vdims']]
            sum_agg = ds.summary(**{str(params['vdims'][0]): agg_fn, 'index': ds.where(sel_fn)})
            agg = self._apply_datashader(dfdata, cvs_fn, sum_agg, agg_kwargs, x, y)
            _ignore = [*params['vdims'], 'index']
            sel_vdims = [s for s in agg if s not in _ignore]
            params['vdims'] = [*params['vdims'], *sel_vdims]
        else:
            agg = self._apply_datashader(dfdata, cvs_fn, agg_fn, agg_kwargs, x, y)
        if 'x_axis' in agg.coords and 'y_axis' in agg.coords:
            agg = agg.rename({'x_axis': x, 'y_axis': y})
        if xtype == 'datetime':
            agg[x.name] = agg[x.name].astype('datetime64[ns]')
        if ytype == 'datetime':
            agg[y.name] = agg[y.name].astype('datetime64[ns]')
        if isinstance(agg, xr.Dataset) or agg.ndim == 2:
            eldata = agg if ds_version > Version('0.5.0') else (xs, ys, agg.data)
            return self.p.element_type(eldata, **params)
        else:
            params['vdims'] = list(agg.coords[agg_fn.column].data)
            return ImageStack(agg, **params)

    def _apply_datashader(self, dfdata, cvs_fn, agg_fn, agg_kwargs, x, y):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='casting datetime64', category=FutureWarning)
            agg = cvs_fn(dfdata, x.name, y.name, agg_fn, **agg_kwargs)
        is_where_index = ds15 and isinstance(agg_fn, ds.where) and isinstance(agg_fn.column, rd.SpecialColumn)
        is_summary_index = isinstance(agg_fn, ds.summary) and 'index' in agg
        if is_where_index or is_summary_index:
            if is_where_index:
                data = agg.data
                agg = agg.to_dataset(name='index')
            else:
                data = agg.index.data
            neg1 = data == -1
            for col in dfdata.columns:
                if col in agg.coords:
                    continue
                val = dfdata[col].values[data]
                if val.dtype.kind == 'f':
                    val[neg1] = np.nan
                elif isinstance(val.dtype, pd.CategoricalDtype):
                    val = val.to_numpy()
                    val[neg1] = '-'
                elif val.dtype.kind == 'O':
                    val[neg1] = '-'
                elif val.dtype.kind == 'M':
                    val[neg1] = np.datetime64('NaT')
                else:
                    val = val.astype(np.float64)
                    val[neg1] = np.nan
                agg[col] = ((y.name, x.name), val)
        return agg