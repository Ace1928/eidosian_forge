import xml.etree.ElementTree as ET
import rasterio
from rasterio._warp import WarpedVRTReaderBase
from rasterio.dtypes import _gdal_typename
from rasterio.enums import MaskFlags
from rasterio._path import _parse_path
from rasterio.transform import TransformMethodsMixin
from rasterio.windows import WindowMethodsMixin
Make a VRT XML document.

    Parameters
    ----------
    src_dataset : Dataset
        The dataset to wrap.
    background : int or float, optional
        The background fill value for the boundless VRT.
    masked : bool
        If True, the src_dataset is replaced by its valid data mask.

    Returns
    -------
    str
        An XML text string.
    