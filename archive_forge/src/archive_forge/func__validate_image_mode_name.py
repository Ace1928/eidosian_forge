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
def _validate_image_mode_name(self, a):
    """
        Validate the image_mode passed in.

        Needed because image_mode used elsewhere in ParlAI is not always consistent with
        what the image teacher allows.
        """
    if not isinstance(a, str):
        raise argparse.ArgumentTypeError('%s must be a string representing image model name' % a)
    available_model_names = self.get_available_image_mode_names()
    if a not in available_model_names:
        raise argparse.ArgumentTypeError('"%s" unknown image model name. Choose from: %s. Currently suggested resnet is resnet152 and resnext is resnext101_32x48d_wsl.' % (a, available_model_names))
    return a