import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def test_wrap_unwrap_observation(base_env=envs.MINERL_OBTAIN_DIAMOND_V0, common_envs=None):
    """
    Tests that wrap_observation composed with unwrap observation is the identity.
    1. Construct an VecWrapper of an EnvSpec called ObtainDiamond
    2. Sample observation from its observation space
    3. Wrap and unwrap those observations.
    4. Assert that the result is the same as the sample
    """
    np.random.seed(42)
    vec_env = wrappers.Vectorized(base_env, common_envs)
    s = base_env.observation_space.sample()
    ws = vec_env.wrap_observation(s)
    us = vec_env.unwrap_observation(ws)
    assert_equal_recursive(s, us)