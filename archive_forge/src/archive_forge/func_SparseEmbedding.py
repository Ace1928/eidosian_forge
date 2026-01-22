from ._internal import NDArrayBase
from ..base import _Null
def SparseEmbedding(data=None, weight=None, input_dim=_Null, output_dim=_Null, dtype=_Null, sparse_grad=_Null, out=None, name=None, **kwargs):
    """Maps integer indices to vector representations (embeddings).

    note:: ``contrib.SparseEmbedding`` is deprecated, use ``Embedding`` instead.

    This operator maps words to real-valued vectors in a high-dimensional space,
    called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
    For example, it has been noted that in the learned embedding spaces, similar words tend
    to be close to each other and dissimilar words far apart.

    For an input array of shape (d1, ..., dK),
    the shape of an output array is (d1, ..., dK, output_dim).
    All the input values should be integers in the range [0, input_dim).

    If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
    (ip0, op0).

    The storage type of the gradient will be `row_sparse`.

    .. Note::

        `SparseEmbedding` is designed for the use case where `input_dim` is very large (e.g. 100k).
        The operator is available on both CPU and GPU.
        When `deterministic` is set to `True`, the accumulation of gradients follows a
        deterministic order if a feature appears multiple times in the input. However, the
        accumulation is usually slower when the order is enforced on GPU.
        When the operator is used on the GPU, the recommended value for `deterministic` is `True`.

    Examples::

      input_dim = 4
      output_dim = 5

      // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
      y = [[  0.,   1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.],
           [ 15.,  16.,  17.,  18.,  19.]]

      // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
      x = [[ 1.,  3.],
           [ 0.,  2.]]

      // Mapped input x to its vector representation y.
      SparseEmbedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                                     [ 15.,  16.,  17.,  18.,  19.]],

                                    [[  0.,   1.,   2.,   3.,   4.],
                                     [ 10.,  11.,  12.,  13.,  14.]]]



    Defined in ../src/operator/tensor/indexing_op.cc:L674

    Parameters
    ----------
    data : NDArray
        The input array to the embedding operator.
    weight : NDArray
        The embedding weight matrix.
    input_dim : int, required
        Vocabulary size of the input indices.
    output_dim : int, required
        Dimension of the embedding vectors.
    dtype : {'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Data type of weight.
    sparse_grad : boolean, optional, default=0
        Compute row sparse gradient in the backward calculation. If set to True, the grad's storage type is row_sparse.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)