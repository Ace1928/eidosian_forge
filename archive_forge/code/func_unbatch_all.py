from typing import List
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType, TensorStructType
def unbatch_all(self) -> List[List[TensorType]]:
    """Unbatch both the repeat and batch dimensions into Python lists.

        This is only supported in PyTorch / TF eager mode.

        This lets you view the data unbatched in its original form, but is
        not efficient for processing.

        .. testcode::
            :skipif: True

            batch = RepeatedValues(<Tensor shape=(B, N, K)>)
            items = batch.unbatch_all()
            print(len(items) == B)

        .. testoutput::

            True

        .. testcode::
            :skipif: True

            print(max(len(x) for x in items) <= N)

        .. testoutput::

            True

        .. testcode::
            :skipif: True

            print(items)

        .. testoutput::

            [[<Tensor_1 shape=(K)>, ..., <Tensor_N, shape=(K)>],
             ...
             [<Tensor_1 shape=(K)>, <Tensor_2 shape=(K)>],
             ...
             [<Tensor_1 shape=(K)>],
             ...
             [<Tensor_1 shape=(K)>, ..., <Tensor_N shape=(K)>]]
        """
    if self._unbatched_repr is None:
        B = _get_batch_dim_helper(self.values)
        if B is None:
            raise ValueError('Cannot call unbatch_all() when batch_dim is unknown. This is probably because you are using TF graph mode.')
        else:
            B = int(B)
        slices = self.unbatch_repeat_dim()
        result = []
        for i in range(B):
            if hasattr(self.lengths[i], 'item'):
                dynamic_len = int(self.lengths[i].item())
            else:
                dynamic_len = int(self.lengths[i].numpy())
            dynamic_slice = []
            for j in range(dynamic_len):
                dynamic_slice.append(_batch_index_helper(slices, i, j))
            result.append(dynamic_slice)
        self._unbatched_repr = result
    return self._unbatched_repr