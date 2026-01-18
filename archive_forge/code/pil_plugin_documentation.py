import numpy as np
from PIL import Image
from ...util import img_as_ubyte, img_as_uint
Save an image to disk.

    Parameters
    ----------
    fname : str or file-like object
        Name of destination file.
    arr : ndarray of uint8 or float
        Array (image) to save.  Arrays of data-type uint8 should have
        values in [0, 255], whereas floating-point arrays must be
        in [0, 1].
    format_str: str
        Format to save as, this is defaulted to PNG if using a file-like
        object; this will be derived from the extension if fname is a string
    kwargs: dict
        Keyword arguments to the Pillow save function (or tifffile save
        function, for Tiff files). These are format dependent. For example,
        Pillow's JPEG save function supports an integer ``quality`` argument
        with values in [1, 95], while TIFFFile supports a ``compress``
        integer argument with values in [0, 9].

    Notes
    -----
    Use the Python Imaging Library.
    See PIL docs [1]_ for a list of other supported formats.
    All images besides single channel PNGs are converted using `img_as_uint8`.
    Single Channel PNGs have the following behavior:
    - Integer values in [0, 255] and Boolean types -> img_as_uint8
    - Floating point and other integers -> img_as_uint16

    References
    ----------
    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    