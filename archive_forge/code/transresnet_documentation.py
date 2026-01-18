from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build
import os
import random
import json
import numpy as np
import torch
import tqdm

        Load a model.

        :param path:
            path from which to load model
        