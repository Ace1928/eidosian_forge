from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def _test_batch_map(stackable_space, batch_dims=(32, 16), no_op=False):
    n_in_batch = np.prod(batch_dims).astype(int)
    if no_op:
        batch = stackable_space.no_op(batch_dims)
    else:
        batch = np.array([stackable_space.sample() for _ in range(n_in_batch)])
        batch = batch.reshape(list(batch_dims) + list(batch[0].shape))
    unmapped = stackable_space.unmap(stackable_space.flat_map(batch))
    if unmapped.dtype.type is np.float:
        assert np.allclose(unmapped, batch)
    else:
        assert np.all(unmapped == batch)