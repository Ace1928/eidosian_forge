from collections import namedtuple
from multiprocessing import current_process
import ctypes
import struct
import numbers
import numpy as np
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
def unpack_img(s, iscolor=-1):
    """Unpack a MXImageRecord to image.

    Parameters
    ----------
    s : str
        String buffer from ``MXRecordIO.read``.
    iscolor : int
        Image format option for ``cv2.imdecode``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    img : numpy.ndarray
        Unpacked image.

    Examples
    --------
    >>> record = mx.recordio.MXRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, img = mx.recordio.unpack_img(item)
    >>> header
    HEADER(flag=0, label=14.0, id=20129312, id2=0)
    >>> img
    array([[[ 23,  27,  45],
            [ 28,  32,  50],
            ...,
            [ 36,  40,  59],
            [ 35,  39,  58]],
           ...,
           [[ 91,  92, 113],
            [ 97,  98, 119],
            ...,
            [168, 169, 167],
            [166, 167, 165]]], dtype=uint8)
    """
    header, s = unpack(s)
    img = np.frombuffer(s, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return (header, img)