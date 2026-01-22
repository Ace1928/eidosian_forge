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
def _next_file(self) -> FileType:
    if self.cur_file is None and self.ioctx.worker is not None:
        idx = self.ioctx.worker.worker_index
        total = self.ioctx.worker.num_workers or 1
        path = self.files[round((len(self.files) - 1) * (idx / total))]
    else:
        path = random.choice(self.files)
    return self._try_open_file(path)