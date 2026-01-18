from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import math
import json
import random

Allows a model to self-chat on a given task.
