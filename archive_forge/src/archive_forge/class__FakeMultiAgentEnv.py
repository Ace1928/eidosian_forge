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
class _FakeMultiAgentEnv(_FakeEnvMixin, _MultiAgentEnv):
    """The fake multiagent environment."""
    pass