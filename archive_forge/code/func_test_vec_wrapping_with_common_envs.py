import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def test_vec_wrapping_with_common_envs():
    base_env = envs.MINERL_TREECHOP_V0
    common_env = [envs.MINERL_TREECHOP_V0, envs.MINERL_NAVIGATE_DENSE_V0]
    test_wrap_unwrap_observation(base_env, common_env)
    test_wrap_unwrap_action(base_env, common_env)
    base_env = envs.MINERL_OBTAIN_DIAMOND_V0
    common_env = [envs.MINERL_OBTAIN_DIAMOND_V0, envs.MINERL_NAVIGATE_DENSE_V0]
    test_wrap_unwrap_observation(base_env, common_env)
    test_wrap_unwrap_action(base_env, common_env)
    common_envs = [envs.MINERL_OBTAIN_DIAMOND_V0, envs.MINERL_TREECHOP_V0, envs.MINERL_NAVIGATE_V0, envs.MINERL_OBTAIN_IRON_PICKAXE_V0]
    for base_env in common_envs:
        test_wrap_unwrap_observation(base_env, common_envs)
        test_wrap_unwrap_action(base_env, common_envs)