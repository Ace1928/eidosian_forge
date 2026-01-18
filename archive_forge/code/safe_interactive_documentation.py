from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.safe_local_human.safe_local_human import SafeLocalHumanAgent
import parlai.utils.logging as logging
import random

Script for safety protected interaction between a local human keyboard input and a
trained model.
