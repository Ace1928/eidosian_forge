from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_unmap_flat_map_enum():
    d = Enum('type1', 'type2')
    x = d.sample()
    assert np.array_equal(d.unmap(d.flat_map(x)), x)