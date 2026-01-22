import glob
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import tree  # pip install dm_tree
from typing import List, Optional, TYPE_CHECKING, Union
from urllib.parse import urlparse
import zipfile
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.spaces.space_utils import clip_action, normalize_action
from ray.rllib.utils.typing import Any, FileType, SampleBatchType
@DeveloperAPI
def _adjust_dones(json_data: dict) -> dict:
    """Make sure DONES in json data is properly translated into TERMINATEDS."""
    new_json_data = {}
    for k, v in json_data.items():
        if k == SampleBatch.DONES:
            new_json_data[SampleBatch.TERMINATEDS] = v
        else:
            new_json_data[k] = v
    return new_json_data