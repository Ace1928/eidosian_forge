import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def test_obf_wrapper(base_env=envs.MINERL_OBTAIN_DIAMOND_V0, common_envs=[envs.MINERL_NAVIGATE_DENSE_V0, envs.MINERL_OBTAIN_DIAMOND_V0]):
    """
    Tests that wrap_action composed with unwrap action is the identity.
    1. Construct an VecWrapper of an EnvSpec called ObtainDiamond
    2. Sample actions from its action space
    3. Wrap and unwrap those actions.
    4. Assert that the result is the same as the sample
    """
    vec_env = envs.MINERL_OBTAIN_DIAMOND_OBF_V0
    base_env.action_space.seed(1)
    base_env.observation_space.seed(1)
    vec_env.action_space.seed(1)
    vec_env.observation_space.seed(1)
    for _ in range(100):
        s = base_env.action_space.sample()
        ws = vec_env.wrap_action(s)
        us = vec_env.unwrap_action(ws)
        assert_equal_recursive(s, us)
        s = vec_env.action_space.sample()
        us = vec_env.unwrap_action(s)
        assert us in base_env.action_space
    for _ in range(100):
        s = base_env.observation_space.sample()
        ws = vec_env.wrap_observation(s)
        us = vec_env.unwrap_observation(ws)
        assert_equal_recursive(s, us, atol=20)
        s = vec_env.observation_space.sample()
        us = vec_env.unwrap_observation(s)
        assert us in base_env.observation_space