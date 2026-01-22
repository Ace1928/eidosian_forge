from ._internal import NDArrayBase
from ..base import _Null
def Reshape(data=None, shape=_Null, reverse=_Null, target_shape=_Null, keep_highest=_Null, out=None, name=None, **kwargs):
    """Reshapes the input array.
    .. note:: ``Reshape`` is deprecated, use ``reshape``
    Given an array and a shape, this function returns a copy of the array in the new shape.
    The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.
    Example::
      reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
    Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:
    - ``0``  copy this dimension from the input to the output shape.
      Example::
      - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
      - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
    - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
      keeping the size of the new array same as that of the input array.
      At most one dimension of shape can be -1.
      Example::
      - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
      - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
      - input shape = (2,3,4), shape=(-1,), output shape = (24,)
    - ``-2`` copy all/remainder of the input dimensions to the output shape.
      Example::
      - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
      - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
      - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
    - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.
      Example::
      - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
      - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
      - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
      - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
    - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
      Example::
      - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
      - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
    If the argument `reverse` is set to 1, then the special values are inferred from right to left.
      Example::
      - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
      - with reverse=1, output shape will be (50,4).


    Defined in ../src/operator/tensor/matrix_op.cc:L174

    Parameters
    ----------
    data : NDArray
        Input data to reshape.
    shape : Shape(tuple), optional, default=[]
        The target shape
    reverse : boolean, optional, default=0
        If true then the special values are inferred from right to left
    target_shape : Shape(tuple), optional, default=[]
        (Deprecated! Use ``shape`` instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims
    keep_highest : boolean, optional, default=0
        (Deprecated! Use ``shape`` instead.) Whether keep the highest dim unchanged.If set to true, then the first dim in target_shape is ignored,and always fixed as input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Examples
    --------
    Reshapes the input array into a new shape.

    >>> x = mx.nd.array([1, 2, 3, 4])
    >>> y = mx.nd.reshape(x, shape=(2, 2))
    >>> x.shape
    (4L,)
    >>> y.shape
    (2L, 2L)
    >>> y.asnumpy()
    array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)

    You can use ``0`` to copy a particular dimension from the input to the output shape
    and '-1' to infer the dimensions of the output.

    >>> x = mx.nd.ones((2, 3, 4))
    >>> x.shape
    (2L, 3L, 4L)
    >>> y = mx.nd.reshape(x, shape=(4, 0, -1))
    >>> y.shape
    (4L, 3L, 2L)
    """
    return (0,)