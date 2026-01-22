from ._internal import NDArrayBase
from ..base import _Null
def MultiBoxPrior(data=None, sizes=_Null, ratios=_Null, clip=_Null, steps=_Null, offsets=_Null, out=None, name=None, **kwargs):
    """Generate prior(anchor) boxes from data, sizes and ratios.

    Parameters
    ----------
    data : NDArray
        Input data.
    sizes : tuple of <float>, optional, default=[1]
        List of sizes of generated MultiBoxPriores.
    ratios : tuple of <float>, optional, default=[1]
        List of aspect ratios of generated MultiBoxPriores.
    clip : boolean, optional, default=0
        Whether to clip out-of-boundary boxes.
    steps : tuple of <float>, optional, default=[-1,-1]
        Priorbox step across y and x, -1 for auto calculation.
    offsets : tuple of <float>, optional, default=[0.5,0.5]
        Priorbox center offsets, y and x respectively

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)