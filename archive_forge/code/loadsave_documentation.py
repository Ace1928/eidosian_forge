from __future__ import annotations
import os
import typing as ty
import numpy as np
from .arrayproxy import is_proxy
from .deprecated import deprecate_with_version
from .filebasedimages import ImageFileError
from .filename_parser import _stringify_path, splitext_addext
from .imageclasses import all_image_classes
from .openers import ImageOpener
Read data from image associated with files

    If you want unscaled data, please use ``img.dataobj.get_unscaled()``
    instead.  If you want scaled data, use ``img.get_fdata()`` (which will cache
    the loaded array) or ``np.array(img.dataobj)`` (which won't cache the
    array). If you want to load the data as for a modified header, save the
    image with the modified header, and reload.

    Parameters
    ----------
    img : ``SpatialImage``
       Image with valid image file in ``img.file_map``.  Unlike the
       ``img.get_fdata()`` method, this function returns the data read
       from the image file, as specified by the *current* image header
       and *current* image files.
    prefer : str, optional
       Can be 'scaled' - in which case we return the data with the
       scaling suggested by the format, or 'unscaled', in which case we
       return, if we can, the raw data from the image file, without the
       scaling applied.

    Returns
    -------
    arr : ndarray
       array as read from file, given parameters in header

    Notes
    -----
    Summary: please use the ``get_data`` method of `img` instead of this
    function unless you are sure what you are doing.

    In general, you will probably prefer ``prefer='scaled'``, because
    this gives the data as the image format expects to return it.

    Use `prefer` == 'unscaled' with care; the modified Analyze-type
    formats such as SPM formats, and nifti1, specify that the image data
    array is given by the raw data on disk, multiplied by a scalefactor
    and maybe with the addition of a constant.  This function, with
    ``unscaled`` returns the data on the disk, without these
    format-specific scalings applied.  Please use this function only if
    you absolutely need the unscaled data, and the magnitude of the
    data, as given by the scalefactor, is not relevant to your
    application.  The Analyze-type formats have a single scalefactor +/-
    offset per image on disk. If you do not care about the absolute
    values, and will be removing the mean from the data, then the
    unscaled values will have preserved intensity ratios compared to the
    mean-centered scaled data.  However, this is not necessarily true of
    other formats with more complicated scaling - such as MINC.
    