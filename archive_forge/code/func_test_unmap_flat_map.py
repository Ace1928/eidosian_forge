from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_unmap_flat_map():
    md = MultiDiscrete([3, 4])
    x = md.sample()
    assert np.array_equal(md.unmap(md.flat_map(x)), x)