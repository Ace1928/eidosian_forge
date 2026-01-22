import enum
from dataclasses import dataclass, field
from typing import Optional, Union
from peft.config import PromptLearningConfig
from peft.utils import PeftType
class PromptTuningInit(str, enum.Enum):
    TEXT = 'TEXT'
    RANDOM = 'RANDOM'