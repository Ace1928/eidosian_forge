import warnings
import numpy as np
import pandas as pd
from pandas.plotting import PlotAccessor
from pandas import CategoricalDtype
import geopandas
from packaging.version import Version
from ._decorator import doc
def _sanitize_geoms(geoms, prefix='Multi'):
    """
    Returns Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry. At the same time, empty or missing geometries are
    filtered out.  Maintains 1:1 matching of geometry to value.

    Prefix specifies type of geometry to be flatten. 'Multi' for MultiPoint and similar,
    "Geom" for GeometryCollection.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    components, component_index = ([], [])
    if not geoms.geom_type.str.startswith(prefix).any() and (not geoms.is_empty.any()) and (not geoms.isna().any()):
        return (geoms, np.arange(len(geoms)))
    for ix, geom in enumerate(geoms):
        if geom is not None and geom.geom_type.startswith(prefix) and (not geom.is_empty):
            for poly in geom.geoms:
                components.append(poly)
                component_index.append(ix)
        elif geom is None or geom.is_empty:
            continue
        else:
            components.append(geom)
            component_index.append(ix)
    return (components, np.array(component_index))