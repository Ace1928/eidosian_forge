from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def applyTransform(im, transform, inPlace=False):
    """
    (pyCMS) Applies a transform to a given image.

    If ``im.mode != transform.inMode``, a :exc:`PyCMSError` is raised.

    If ``inPlace`` is ``True`` and ``transform.inMode != transform.outMode``, a
    :exc:`PyCMSError` is raised.

    If ``im.mode``, ``transform.inMode`` or ``transform.outMode`` is not
    supported by pyCMSdll or the profiles you used for the transform, a
    :exc:`PyCMSError` is raised.

    If an error occurs while the transform is being applied,
    a :exc:`PyCMSError` is raised.

    This function applies a pre-calculated transform (from
    ImageCms.buildTransform() or ImageCms.buildTransformFromOpenProfiles())
    to an image. The transform can be used for multiple images, saving
    considerable calculation time if doing the same conversion multiple times.

    If you want to modify im in-place instead of receiving a new image as
    the return value, set ``inPlace`` to ``True``.  This can only be done if
    ``transform.inMode`` and ``transform.outMode`` are the same, because we can't
    change the mode in-place (the buffer sizes for some modes are
    different).  The default behavior is to return a new :py:class:`~PIL.Image.Image`
    object of the same dimensions in mode ``transform.outMode``.

    :param im: An :py:class:`~PIL.Image.Image` object, and im.mode must be the same
        as the ``inMode`` supported by the transform.
    :param transform: A valid CmsTransform class object
    :param inPlace: Bool.  If ``True``, ``im`` is modified in place and ``None`` is
        returned, if ``False``, a new :py:class:`~PIL.Image.Image` object with the
        transform applied is returned (and ``im`` is not changed). The default is
        ``False``.
    :returns: Either ``None``, or a new :py:class:`~PIL.Image.Image` object,
        depending on the value of ``inPlace``. The profile will be returned in
        the image's ``info['icc_profile']``.
    :exception PyCMSError:
    """
    try:
        if inPlace:
            transform.apply_in_place(im)
            imOut = None
        else:
            imOut = transform.apply(im)
    except (TypeError, ValueError) as v:
        raise PyCMSError(v) from v
    return imOut