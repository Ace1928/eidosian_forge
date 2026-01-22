from ._internal import NDArrayBase
from ..base import _Null
def _cvimresize(src=None, w=_Null, h=_Null, interp=_Null, out=None, name=None, **kwargs):
    """Resize image with OpenCV. 


    Parameters
    ----------
    src : NDArray
        source image
    w : int, required
        Width of resized image.
    h : int, required
        Height of resized image.
    interp : int, optional, default='1'
        Interpolation method (default=cv2.INTER_LINEAR).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)