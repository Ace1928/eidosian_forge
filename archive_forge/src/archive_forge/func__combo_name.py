from parlai import __file__ as parlai_filepath
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args as acute_add_args
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.conversations import Conversations, Conversation
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output
from parlai.mturk.tasks.acute_eval.analysis import (
from parlai.mturk.tasks.acute_eval.dump_task_to_acute_format import (
from parlai.mturk.tasks.acute_eval.configs import CONFIG
from typing import Dict, Any, List, Tuple, Set
from itertools import combinations
import datetime
import time
import json
import os
import random
import torch
import hashlib
def _combo_name(id1, id2):
    """
            Return joined name for combo of comparisons.
            """
    id1_name = id1
    id2_name = id2
    if 'model' in CONFIG[id1]:
        id1_name += self.task.replace(':', '_')
    if 'model' in CONFIG[id2]:
        id2_name += self.task.replace(':', '_')
    return f'{id1_name}__vs__{id2_name}'