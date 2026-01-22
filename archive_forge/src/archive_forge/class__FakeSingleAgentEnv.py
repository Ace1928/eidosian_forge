import copy
import logging
import os
from typing import Any, Dict, Tuple
from lxml import etree
import json
import numpy as np
from minerl.env._multiagent import _MultiAgentEnv
from minerl.env._singleagent import _SingleAgentEnv
from minerl.herobraine.env_specs.navigate_specs import Navigate
class _FakeSingleAgentEnv(_FakeEnvMixin, _SingleAgentEnv):
    """The fake singleagent environment."""

    def step(self, action):
        aname = self.task.agent_names[0]
        multi_agent_action = {aname: action}
        s, reward, done, info = super().step(multi_agent_action)
        return (s[aname], reward[aname], done, info[aname])