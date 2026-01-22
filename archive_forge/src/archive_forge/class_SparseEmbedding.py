import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm
class SparseEmbedding(Block):
    """Turns non-negative integers (indexes/tokens) into dense vectors
    of fixed size. eg. [4, 20] -> [[0.25, 0.1], [0.6, -0.2]]

    This SparseBlock is designed for distributed training with extremely large
    input dimension. Both weight and gradient w.r.t. weight are `RowSparseNDArray`.

    Note: if `sparse_grad` is set to True, the gradient w.r.t weight will be
    sparse. Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
    and Adam. By default lazy updates is turned on, which may perform differently
    from standard updates. For more details, please check the Optimization API at:
    https://mxnet.incubator.apache.org/api/python/optimization/optimization.html

    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : Initializer
        Initializer for the `embeddings` matrix.

    Inputs:
        - **data**: (N-1)-D tensor with shape: `(x1, x2, ..., xN-1)`.
    Output:
        - **out**: N-D tensor with shape: `(x1, x2, ..., xN-1, output_dim)`.
    """

    def __init__(self, input_dim, output_dim, dtype='float32', weight_initializer=None, **kwargs):
        super(SparseEmbedding, self).__init__(**kwargs)
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim, 'dtype': dtype, 'sparse_grad': True}
        self.weight = self.params.get('weight', shape=(input_dim, output_dim), init=weight_initializer, dtype=dtype, grad_stype='row_sparse', stype='row_sparse')

    def forward(self, x):
        weight = self.weight.row_sparse_data(x)
        return nd.Embedding(x, weight, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{block_name}({input_dim} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__, **self._kwargs)