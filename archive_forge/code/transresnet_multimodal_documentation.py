from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from parlai.core.message import Message
from .modules import TransresnetMultimodalModel
from projects.personality_captions.transresnet.transresnet import TransresnetAgent
import torch
from torch import optim
import random
import os
import numpy as np
import tqdm
from collections import deque

        Report per-dialogue round metrics.
        