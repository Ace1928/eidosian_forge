from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from controllable_seq2seq.controls import (
from controllable_seq2seq.util import ConvAI2History
from collections import Counter
import copy
import random
import json
import time
import os

    Evaluates a model.

    :param opt: tells the evaluation function how to run
    