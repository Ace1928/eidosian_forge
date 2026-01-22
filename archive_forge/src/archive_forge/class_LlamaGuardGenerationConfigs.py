import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
@dataclass
class LlamaGuardGenerationConfigs:
    should_list_violated_codes: bool
    explanation_position: Optional[ExplanationPosition]