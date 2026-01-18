import abc
import copy
from collections import OrderedDict
from minerl.herobraine.env_spec import EnvSpec
import minerl
def unwrap_observation(self, obs: OrderedDict) -> OrderedDict:
    obs = copy.deepcopy(obs)
    if minerl.utils.test.SHOULD_ASSERT:
        assert obs in self.observation_space
    obs = self._unwrap_observation(obs)
    if minerl.utils.test.SHOULD_ASSERT:
        assert obs in self.env_to_wrap.observation_space
    if self._unwrap_obs_fn is not None:
        obs = self._unwrap_obs_fn(obs)
    return obs