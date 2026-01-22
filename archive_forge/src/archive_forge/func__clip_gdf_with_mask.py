import warnings
import numpy as np
import pandas.api.types
from shapely.geometry import Polygon, MultiPolygon, box
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _clip_gdf_with_mask(gdf, mask):
    """Clip geometry to the polygon/rectangle extent.

    Clip an input GeoDataFrame to the polygon extent of the polygon
    parameter.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Dataframe to clip.

    mask : (Multi)Polygon, list-like
        Reference polygon/rectangle for clipping.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with polygon/rectangle.
    """
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    if clipping_by_rectangle:
        intersection_polygon = box(*mask)
    else:
        intersection_polygon = mask
    gdf_sub = gdf.iloc[gdf.sindex.query(intersection_polygon, predicate='intersects')]
    non_point_mask = gdf_sub.geom_type != 'Point'
    if not non_point_mask.any():
        return gdf_sub
    if isinstance(gdf_sub, GeoDataFrame):
        clipped = gdf_sub.copy()
        if clipping_by_rectangle:
            clipped.loc[non_point_mask, clipped._geometry_column_name] = gdf_sub.geometry.values[non_point_mask].clip_by_rect(*mask)
        else:
            clipped.loc[non_point_mask, clipped._geometry_column_name] = gdf_sub.geometry.values[non_point_mask].intersection(mask)
    else:
        clipped = gdf_sub.copy()
        if clipping_by_rectangle:
            clipped[non_point_mask] = gdf_sub.values[non_point_mask].clip_by_rect(*mask)
        else:
            clipped[non_point_mask] = gdf_sub.values[non_point_mask].intersection(mask)
    if clipping_by_rectangle:
        clipped = clipped[~clipped.is_empty]
    return clipped