import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_phonemizer_available, logging
Converts an index (integer) in a token (str) using the vocab.