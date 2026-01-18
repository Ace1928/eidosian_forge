import json
import os
from typing import Dict, List, Optional, Tuple
import regex as re
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

        A very simple chat template that just adds whitespace between messages.
        