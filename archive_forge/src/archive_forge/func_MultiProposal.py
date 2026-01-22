from ._internal import NDArrayBase
from ..base import _Null
def MultiProposal(cls_prob=None, bbox_pred=None, im_info=None, rpn_pre_nms_top_n=_Null, rpn_post_nms_top_n=_Null, threshold=_Null, rpn_min_size=_Null, scales=_Null, ratios=_Null, feature_stride=_Null, output_score=_Null, iou_loss=_Null, out=None, name=None, **kwargs):
    """Generate region proposals via RPN

    Parameters
    ----------
    cls_prob : NDArray
        Score of how likely proposal is object.
    bbox_pred : NDArray
        BBox Predicted deltas from anchors for proposals
    im_info : NDArray
        Image size and scale.
    rpn_pre_nms_top_n : int, optional, default='6000'
        Number of top scoring boxes to keep before applying NMS to RPN proposals
    rpn_post_nms_top_n : int, optional, default='300'
        Number of top scoring boxes to keep after applying NMS to RPN proposals
    threshold : float, optional, default=0.699999988
        NMS value, below which to suppress.
    rpn_min_size : int, optional, default='16'
        Minimum height or width in proposal
    scales : tuple of <float>, optional, default=[4,8,16,32]
        Used to generate anchor windows by enumerating scales
    ratios : tuple of <float>, optional, default=[0.5,1,2]
        Used to generate anchor windows by enumerating ratios
    feature_stride : int, optional, default='16'
        The size of the receptive field each unit in the convolution layer of the rpn,for example the product of all stride's prior to this layer.
    output_score : boolean, optional, default=0
        Add score to outputs
    iou_loss : boolean, optional, default=0
        Usage of IoU Loss

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)