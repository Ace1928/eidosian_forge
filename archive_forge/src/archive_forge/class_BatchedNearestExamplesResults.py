import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
class BatchedNearestExamplesResults(NamedTuple):
    total_scores: List[List[float]]
    total_examples: List[dict]