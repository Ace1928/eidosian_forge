import numpy as np
import ray
import ray.experimental.array.remote as ra
class DistArray:

    def __init__(self, shape, object_refs=None):
        self.shape = shape
        self.ndim = len(shape)
        self.num_blocks = [int(np.ceil(1.0 * a / BLOCK_SIZE)) for a in self.shape]
        if object_refs is not None:
            self.object_refs = object_refs
        else:
            self.object_refs = np.empty(self.num_blocks, dtype=object)
        if self.num_blocks != list(self.object_refs.shape):
            raise Exception('The fields `num_blocks` and `object_refs` are inconsistent, `num_blocks` is {} and `object_refs` has shape {}'.format(self.num_blocks, list(self.object_refs.shape)))

    @staticmethod
    def compute_block_lower(index, shape):
        if len(index) != len(shape):
            raise Exception('The fields `index` and `shape` must have the same length, but `index` is {} and `shape` is {}.'.format(index, shape))
        return [elem * BLOCK_SIZE for elem in index]

    @staticmethod
    def compute_block_upper(index, shape):
        if len(index) != len(shape):
            raise Exception('The fields `index` and `shape` must have the same length, but `index` is {} and `shape` is {}.'.format(index, shape))
        upper = []
        for i in range(len(shape)):
            upper.append(min((index[i] + 1) * BLOCK_SIZE, shape[i]))
        return upper

    @staticmethod
    def compute_block_shape(index, shape):
        lower = DistArray.compute_block_lower(index, shape)
        upper = DistArray.compute_block_upper(index, shape)
        return [u - l for l, u in zip(lower, upper)]

    @staticmethod
    def compute_num_blocks(shape):
        return [int(np.ceil(1.0 * a / BLOCK_SIZE)) for a in shape]

    def assemble(self):
        """Assemble an array from a distributed array of object refs."""
        first_block = ray.get(self.object_refs[(0,) * self.ndim])
        dtype = first_block.dtype
        result = np.zeros(self.shape, dtype=dtype)
        for index in np.ndindex(*self.num_blocks):
            lower = DistArray.compute_block_lower(index, self.shape)
            upper = DistArray.compute_block_upper(index, self.shape)
            value = ray.get(self.object_refs[index])
            result[tuple((slice(l, u) for l, u in zip(lower, upper)))] = value
        return result

    def __getitem__(self, sliced):
        a = self.assemble()
        return a[sliced]