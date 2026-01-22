from ._internal import NDArrayBase
from ..base import _Null
def DeformablePSROIPooling(data=None, rois=None, trans=None, spatial_scale=_Null, output_dim=_Null, group_size=_Null, pooled_size=_Null, part_size=_Null, sample_per_part=_Null, trans_std=_Null, no_trans=_Null, out=None, name=None, **kwargs):
    """Performs deformable position-sensitive region-of-interest pooling on inputs.
    The DeformablePSROIPooling operation is described in https://arxiv.org/abs/1703.06211 .batch_size will change to the number of region bounding boxes after DeformablePSROIPooling

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator, a 4D Feature maps
    rois : Symbol
        Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data
    trans : Symbol
        transition parameter
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
    output_dim : int, required
        fix output dim
    group_size : int, required
        fix group size
    pooled_size : int, required
        fix pooled size
    part_size : int, optional, default='0'
        fix part size
    sample_per_part : int, optional, default='1'
        fix samples per part
    trans_std : float, optional, default=0
        fix transition std
    no_trans : boolean, optional, default=0
        Whether to disable trans parameter.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)