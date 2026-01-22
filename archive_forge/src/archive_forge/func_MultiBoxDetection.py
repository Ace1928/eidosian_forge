from ._internal import NDArrayBase
from ..base import _Null
def MultiBoxDetection(cls_prob=None, loc_pred=None, anchor=None, clip=_Null, threshold=_Null, background_id=_Null, nms_threshold=_Null, force_suppress=_Null, variances=_Null, nms_topk=_Null, out=None, name=None, **kwargs):
    """Convert multibox detection predictions.

    Parameters
    ----------
    cls_prob : NDArray
        Class probabilities.
    loc_pred : NDArray
        Location regression predictions.
    anchor : NDArray
        Multibox prior anchor boxes
    clip : boolean, optional, default=1
        Clip out-of-boundary boxes.
    threshold : float, optional, default=0.00999999978
        Threshold to be a positive prediction.
    background_id : int, optional, default='0'
        Background id.
    nms_threshold : float, optional, default=0.5
        Non-maximum suppression threshold.
    force_suppress : boolean, optional, default=0
        Suppress all detections regardless of class_id.
    variances : tuple of <float>, optional, default=[0.1,0.1,0.2,0.2]
        Variances to be decoded from box regression output.
    nms_topk : int, optional, default='-1'
        Keep maximum top k detections before nms, -1 for no limit.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)