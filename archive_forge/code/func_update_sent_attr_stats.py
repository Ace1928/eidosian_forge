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
def update_sent_attr_stats(sent_attrs, history, prediction):
    """
    Update the sent_attrs dict with the attributes of a prediction with given history.

    Inputs:
      sent_attrs: dictionary mapping each attr (a string) to a list of floats
        (the scores).
      history: a ConvAI2History
      prediction: string. the response text for which we measure sent attributes
    """
    for attr in sent_attrs.keys():
        attr_score = eval_attr(prediction, history, attr)
        sent_attrs[attr].append(attr_score)
    return sent_attrs