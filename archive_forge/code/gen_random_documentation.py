from ._internal import NDArrayBase
from ..base import _Null
Draw random samples from a uniform distribution according to the input array shape.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Example::

       uniform(low=0, high=1, data=ones(2,2)) = [[ 0.60276335,  0.85794562],
                                                 [ 0.54488319,  0.84725171]]



    Defined in ../src/operator/random/sample_op.cc:L208

    Parameters
    ----------
    low : float, optional, default=0
        Lower bound of the distribution.
    high : float, optional, default=1
        Upper bound of the distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    