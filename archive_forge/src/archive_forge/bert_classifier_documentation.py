from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent
from parlai.agents.bert_ranker.helpers import (
from parlai.core.torch_agent import History
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.utils.misc import warn_once
from parlai.zoo.bert.build import download
from collections import deque
import os
import torch

        Score the batch.
        