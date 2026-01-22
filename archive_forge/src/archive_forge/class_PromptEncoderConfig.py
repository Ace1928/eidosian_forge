import enum
from dataclasses import dataclass, field
from typing import Union
from peft.config import PromptLearningConfig
from peft.utils import PeftType
@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEncoder`].

    Args:
        encoder_reparameterization_type (Union[[`PromptEncoderReparameterizationType`], `str`]):
            The type of reparameterization to use.
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        encoder_num_layers (`int`): The number of layers of the prompt encoder.
        encoder_dropout (`float`): The dropout probability of the prompt encoder.
    """
    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(default=PromptEncoderReparameterizationType.MLP, metadata={'help': 'How to reparameterize the prompt encoder'})
    encoder_hidden_size: int = field(default=None, metadata={'help': 'The hidden size of the prompt encoder'})
    encoder_num_layers: int = field(default=2, metadata={'help': 'The number of layers of the prompt encoder'})
    encoder_dropout: float = field(default=0.0, metadata={'help': 'The dropout of the prompt encoder'})

    def __post_init__(self):
        self.peft_type = PeftType.P_TUNING