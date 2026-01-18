import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def to_spatialpandas(data, xdim, ydim, columns=None, geom='point'):
    """Converts list of dictionary format geometries to spatialpandas line geometries.

    Args:
        data: List of dictionaries representing individual geometries
        xdim: Name of x-coordinates column
        ydim: Name of y-coordinates column
        columns: List of columns to add
        geom: The type of geometry

    Returns:
        A spatialpandas.GeoDataFrame version of the data
    """
    from spatialpandas import GeoDataFrame, GeoSeries
    from spatialpandas.geometry import Line, LineArray, MultiLineArray, MultiPointArray, MultiPolygonArray, Point, PointArray, Polygon, PolygonArray, Ring, RingArray
    from ...element import Polygons
    if columns is None:
        columns = []
    poly = any((Polygons._hole_key in d for d in data)) or geom == 'Polygon'
    if poly:
        geom_type = Polygon
        single_array, multi_array = (PolygonArray, MultiPolygonArray)
    elif geom == 'Line':
        geom_type = Line
        single_array, multi_array = (LineArray, MultiLineArray)
    elif geom == 'Ring':
        geom_type = Ring
        single_array, multi_array = (RingArray, MultiLineArray)
    else:
        geom_type = Point
        single_array, multi_array = (PointArray, MultiPointArray)
    array_type = None
    hole_arrays, geom_arrays = ([], [])
    for geom in data:
        geom = dict(geom)
        if xdim not in geom or ydim not in geom:
            raise ValueError('Could not find geometry dimensions')
        xs, ys = (geom.pop(xdim), geom.pop(ydim))
        xscalar, yscalar = (isscalar(xs), isscalar(ys))
        if xscalar and yscalar:
            xs, ys = (np.array([xs]), np.array([ys]))
        elif xscalar:
            xs = np.full_like(ys, xs)
        elif yscalar:
            ys = np.full_like(xs, ys)
        geom_array = np.column_stack([xs, ys])
        if geom_type in (Polygon, Ring):
            geom_array = ensure_ring(geom_array)
        splits = np.where(np.isnan(geom_array[:, :2].astype('float')).sum(axis=1))[0]
        split_geoms = np.split(geom_array, splits + 1) if len(splits) else [geom_array]
        split_holes = geom.pop(Polygons._hole_key, None)
        if split_holes is not None:
            if len(split_holes) != len(split_geoms):
                raise DataError('Polygons with holes containing multi-geometries must declare a list of holes for each geometry.', SpatialPandasInterface)
            else:
                split_holes = [[ensure_ring(np.asarray(h)) for h in hs] for hs in split_holes]
        geom_arrays.append(split_geoms)
        hole_arrays.append(split_holes)
        if geom_type is Point:
            if len(splits) > 1 or any((len(g) > 1 for g in split_geoms)):
                array_type = multi_array
            elif array_type is None:
                array_type = single_array
        elif len(splits):
            array_type = multi_array
        elif array_type is None:
            array_type = single_array
    converted = defaultdict(list)
    for geom, arrays, holes in zip(data, geom_arrays, hole_arrays):
        parts = []
        for i, g in enumerate(arrays):
            if i != len(arrays) - 1:
                g = g[:-1]
            if len(g) < (3 if poly else 2) and geom_type is not Point:
                continue
            if poly:
                parts.append([])
                subparts = parts[-1]
            else:
                subparts = parts
            subparts.append(g[:, :2])
            if poly and holes is not None:
                subparts += [np.array(h) for h in holes[i]]
        for c, v in geom.items():
            converted[c].append(v)
        if array_type is PointArray:
            parts = parts[0].flatten()
        elif array_type is MultiPointArray:
            parts = np.concatenate([sp.flatten() for sp in parts])
        elif array_type is multi_array:
            parts = [[ssp.flatten() for ssp in sp] if poly else sp.flatten() for sp in parts]
        else:
            parts = [np.asarray(sp).flatten() for sp in parts[0]] if poly else parts[0].flatten()
        converted['geometry'].append(parts)
    if converted:
        geometries = converted['geometry']
        if array_type is PointArray:
            geometries = np.concatenate(geometries)
        geom_array = array_type(geometries)
        if poly:
            geom_array = geom_array.oriented()
        converted['geometry'] = GeoSeries(geom_array)
    else:
        converted['geometry'] = GeoSeries(single_array([]))
    return GeoDataFrame(converted, columns=['geometry'] + columns)