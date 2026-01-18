import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
Truncates a pair of sequences to a maximum sequence length.