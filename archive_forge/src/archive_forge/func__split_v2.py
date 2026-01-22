from ._internal import NDArrayBase
from ..base import _Null
def _split_v2(data=None, indices=_Null, axis=_Null, squeeze_axis=_Null, sections=_Null, out=None, name=None, **kwargs):
    """Splits an array along a particular axis into multiple sub-arrays.
    Example::
       x  = [[[ 1.]
              [ 2.]]
             [[ 3.]
              [ 4.]]
             [[ 5.]
              [ 6.]]]
       x.shape = (3, 2, 1)
       y = split_v2(x, axis=1, indices_or_sections=2) // a list of 2 arrays with shape (3, 1, 1)
       y = [[[ 1.]]
            [[ 3.]]
            [[ 5.]]]
           [[[ 2.]]
            [[ 4.]]
            [[ 6.]]]
       y[0].shape = (3, 1, 1)
       z = split_v2(x, axis=0, indices_or_sections=3) // a list of 3 arrays with shape (1, 2, 1)
       z = [[[ 1.]
             [ 2.]]]
           [[[ 3.]
             [ 4.]]]
           [[[ 5.]
             [ 6.]]]
       z[0].shape = (1, 2, 1)
       w = split_v2(x, axis=0, indices_or_sections=(1,)) // a list of 2 arrays with shape [(1, 2, 1), (2, 2, 1)]
       w = [[[ 1.]
             [ 2.]]]
           [[[3.]
             [4.]]
            [[5.]
             [6.]]]
      w[0].shape = (1, 2, 1)
      w[1].shape = (2, 2, 1)
    `squeeze_axis=True` removes the axis with length 1 from the shapes of the output arrays.
    **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
    along the `axis` which it is split.
    Also `squeeze_axis` can be set to true only if ``input.shape[axis] == indices_or_sections``.
    Example::
       z = split_v2(x, axis=0, indices_or_sections=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
       z = [[ 1.]
            [ 2.]]
           [[ 3.]
            [ 4.]]
           [[ 5.]
            [ 6.]]
       z[0].shape = (2, 1)


    Defined in ../src/operator/tensor/matrix_op.cc:L1087

    Parameters
    ----------
    data : NDArray
        The input
    indices : Shape(tuple), required
        Indices of splits. The elements should denote the boundaries of at which split is performed along the `axis`.
    axis : int, optional, default='1'
        Axis along which to split.
    squeeze_axis : boolean, optional, default=0
        If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.
    sections : int, optional, default='0'
        Number of sections if equally splitted. Default to 0 which means split by indices.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)