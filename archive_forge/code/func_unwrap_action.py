import abc
import copy
from collections import OrderedDict
from minerl.herobraine.env_spec import EnvSpec
import minerl
def unwrap_action(self, act: OrderedDict) -> OrderedDict:
    act = copy.deepcopy(act)
    if minerl.utils.test.SHOULD_ASSERT:
        assert act in self.action_space
    act = self._unwrap_action(act)
    if minerl.utils.test.SHOULD_ASSERT:
        assert act in self.env_to_wrap.action_space
    if self._unwrap_act_fn is not None:
        act = self._unwrap_act_fn(act)
    return act