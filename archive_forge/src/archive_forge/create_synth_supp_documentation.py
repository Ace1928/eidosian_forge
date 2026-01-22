import json
import os
import random
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.projects.self_feeding.utils import Parley

    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :return: the final result of calling report()
    