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
def _build_id_pairs(self):
    """
        Generate self.config_ids and self.combos from self.opt.
        """
    choices = CONFIG.keys()
    combos: Set[Tuple[str, str]] = set()
    ids: Set[str] = set()
    if self.opt['ids'] is None and self.opt['id_pairs'] is None:
        raise RuntimeError('Either --ids or --id-pairs should be set for comparision.')
    if self.opt['id_pairs'] is not None:
        id_pairs = self.opt['id_pairs'].split(',')
        id_pairs = [id_pair.split(':') for id_pair in id_pairs]
        for id_pair in id_pairs:
            combos.add(tuple(sorted((id_pair[0], id_pair[1]))))
            ids |= set(id_pair)
    else:
        ids = set(self.opt['ids'].split(','))
        combos = set(combinations(ids, 2))
    self.config_ids: List[str] = list(ids)
    self.config_ids.sort()
    self.combos: List[Tuple[str, str]] = list(combos)
    self.combos.sort()
    for config_id in self.config_ids:
        if config_id not in choices:
            raise RuntimeError(f'ID {config_id} not specified in the config (`configs.py`).')
    assert len(self.config_ids) > 1, 'Must specify least 2 ids'