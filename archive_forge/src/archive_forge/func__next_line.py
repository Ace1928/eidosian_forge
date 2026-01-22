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
def _next_line(self) -> str:
    if not self.cur_file:
        self.cur_file = self._next_file()
    line = self.cur_file.readline()
    tries = 0
    while not line and tries < 100:
        tries += 1
        if hasattr(self.cur_file, 'close'):
            self.cur_file.close()
        self.cur_file = self._next_file()
        line = self.cur_file.readline()
        if not line:
            logger.debug('Ignoring empty file {}'.format(self.cur_file))
    if not line:
        raise ValueError('Failed to read next line from files: {}'.format(self.files))
    return line