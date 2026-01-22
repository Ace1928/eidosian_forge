from dataclasses import dataclass, field
from peft.config import PromptLearningConfig
from peft.utils import PeftType
@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """
    encoder_hidden_size: int = field(default=None, metadata={'help': 'The hidden size of the encoder'})
    prefix_projection: bool = field(default=False, metadata={'help': 'Whether to project the prefix tokens'})

    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING