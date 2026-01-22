import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
def _get_ep_from_turns(self, xturns, yturns):
    eps = []
    for xturn, yturn in zip(xturns, yturns):
        turn = {}
        turn['text'] = xturn.get('text').strip()
        turn['labels'] = [yturn.get('text').strip()]
        turn['episode_done'] = False
        eps.append(turn)
    if eps:
        eps[-1]['episode_done'] = True
        return eps