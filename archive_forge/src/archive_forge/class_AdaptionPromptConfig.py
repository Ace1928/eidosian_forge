from collections import namedtuple
from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.utils import PeftType
from .utils import llama_compute_query_states
@dataclass
class AdaptionPromptConfig(PeftConfig):
    """Stores the configuration of an [`AdaptionPromptModel`]."""
    target_modules: str = field(default=None, metadata={'help': 'Name of the attention submodules to insert adaption prompts into.'})
    adapter_len: int = field(default=None, metadata={'help': 'Number of adapter tokens to insert'})
    adapter_layers: int = field(default=None, metadata={'help': 'Number of adapter layers (from the top)'})

    def __post_init__(self):
        self.peft_type = PeftType.ADAPTION_PROMPT

    @property
    def is_adaption_prompt(self) -> bool:
        """Return True if this is an adaption prompt config."""
        return True