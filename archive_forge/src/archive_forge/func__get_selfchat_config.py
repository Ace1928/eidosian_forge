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
def _get_selfchat_config(self, config_id: str) -> Dict[str, Any]:
    """
        Return config for selfchat.

        :param config_id:
            config_id string

        :return config:
            dict config for self-chat
        """
    outfile = self._get_selfchat_log_path(config_id)
    config = CONFIG[config_id]
    config.update({'task': self.task, 'outfile': outfile, 'num_self_chats': NUM_SELFCHAT_EXAMPLES, 'selfchat_max_turns': SELFCHAT_MAX_TURNS, 'display_examples': False, 'log_every_n_secs': -1, 'indent': -1})
    return config