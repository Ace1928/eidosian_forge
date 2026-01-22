import skimage.io as io
Load an image extension from a FITS file and return a NumPy array

    Parameters
    ----------
    image_ext : tuple
        FITS extension to load, in the format ``(filename, ext_num)``.
        The FITS ``(extname, extver)`` format is unsupported, since this
        function is not called directly by the user and
        ``imread_collection()`` does the work of figuring out which
        extensions need loading.

    