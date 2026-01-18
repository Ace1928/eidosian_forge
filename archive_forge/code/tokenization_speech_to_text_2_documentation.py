import json
import os
from typing import Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

        Converts a list of output tokens into a single string.
        