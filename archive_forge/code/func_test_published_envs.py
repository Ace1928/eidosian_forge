import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def test_published_envs():
    map_common_space_no_op([envs.MINERL_TREECHOP_OBF_V0, envs.MINERL_NAVIGATE_OBF_V0, envs.MINERL_NAVIGATE_DENSE_OBF_V0, envs.MINERL_NAVIGATE_DENSE_EXTREME_OBF_V0, envs.MINERL_OBTAIN_IRON_PICKAXE_DENSE_OBF_V0, envs.MINERL_OBTAIN_DIAMOND_DENSE_OBF_V0])
    test_wrap_unwrap_observation(envs.MINERL_TREECHOP_V0)
    test_wrap_unwrap_observation(envs.MINERL_NAVIGATE_V0)
    test_wrap_unwrap_observation(envs.MINERL_NAVIGATE_DENSE_V0)
    test_wrap_unwrap_observation(envs.MINERL_NAVIGATE_DENSE_EXTREME_V0)
    test_wrap_unwrap_observation(envs.MINERL_OBTAIN_IRON_PICKAXE_DENSE_V0)
    test_wrap_unwrap_observation(envs.MINERL_OBTAIN_DIAMOND_DENSE_V0)