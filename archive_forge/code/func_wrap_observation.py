import abc
import copy
from collections import OrderedDict
from minerl.herobraine.env_spec import EnvSpec
import minerl
def wrap_observation(self, obs: OrderedDict):
    obs = copy.deepcopy(obs)
    if self._wrap_obs_fn is not None:
        obs = self._wrap_obs_fn(obs)
    if minerl.utils.test.SHOULD_ASSERT:
        assert obs in self.env_to_wrap.observation_space
    wrapped_obs = self._wrap_observation(obs)
    if minerl.utils.test.SHOULD_ASSERT:
        assert wrapped_obs in self.observation_space
    return wrapped_obs