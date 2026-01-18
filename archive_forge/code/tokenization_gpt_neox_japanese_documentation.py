import collections
import json
import os
import re
from typing import Optional, Tuple
import numpy as np
from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging

        A simple chat template that just adds BOS/EOS tokens around messages while discarding role information.
        