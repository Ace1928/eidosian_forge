import abc
import copy
from collections import OrderedDict
from minerl.herobraine.env_spec import EnvSpec
import minerl
def wrap_action(self, act: OrderedDict):
    act = copy.deepcopy(act)
    if self._wrap_act_fn is not None:
        act = self._wrap_act_fn(act)
    if minerl.utils.test.SHOULD_ASSERT:
        assert act in self.env_to_wrap.action_space
    wrapped_act = self._wrap_action(act)
    if minerl.utils.test.SHOULD_ASSERT:
        assert wrapped_act in self.action_space
    return wrapped_act