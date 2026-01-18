import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def test_wrap_unwrap_action_navigate():
    test_wrap_unwrap_action(base_env=envs.MINERL_NAVIGATE_DENSE_EXTREME_V0)