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
def _convert_task_to_conversations(self, config_id: str):
    """
        Convert task data to conversations format.

        :param config_id:
            id in config
        """
    self._print_progress(f'Converting task data to conversations format for {config_id}')
    config = self._get_task_conversion_config(config_id)
    with capture_output():
        parser = convert_task_setup_args()
        parser.set_params(**config)
        opt = parser.parse_args(args=[])
    convert_task_data(opt)