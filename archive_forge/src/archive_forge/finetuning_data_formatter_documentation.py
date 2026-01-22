import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence

    Same as in _maybe_add_example_with_dropped_nonviolated_prompt_categories but we
    also drop all of the violated categories from the llama guard prompt.
    