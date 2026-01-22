import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _is_a_prompt_only_example(training_example: TrainingExample) -> bool:
    return training_example.response == 'N/A'