import enum
from dataclasses import dataclass, field
from typing import Union
from peft.config import PromptLearningConfig
from peft.utils import PeftType
class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = 'MLP'
    LSTM = 'LSTM'