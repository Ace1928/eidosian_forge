from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_unmap_flat_map_dict():
    d = Dict({'a': Box(shape=[3, 2], low=-2, high=2, dtype=np.float32)})
    x = d.sample()
    assert_equal_recursive(d.unmap(d.flat_map(x)), x)