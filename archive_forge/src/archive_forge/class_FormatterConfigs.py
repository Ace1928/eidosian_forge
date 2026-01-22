import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
@dataclass
class FormatterConfigs:
    guidelines: Guidelines
    llama_guard_prompt_configs: LlamaGuardPromptConfigs
    llama_guard_generation_configs: LlamaGuardGenerationConfigs
    augmentation_configs: AugmentationConfigs
    random_seed: int = 42