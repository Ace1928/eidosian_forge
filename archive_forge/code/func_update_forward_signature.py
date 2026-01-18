import inspect
from copy import deepcopy
from functools import update_wrapper
from types import MethodType
from .peft_model import PeftModel
def update_forward_signature(model: PeftModel) -> None:
    """
    Args:
    Updates the forward signature of the PeftModel to include parents class signature
        model (`PeftModel`): Peft model to update the forward signature
    Example:

    ```python
    >>> from transformers import WhisperForConditionalGeneration
    >>> from peft import get_peft_model, LoraConfig, update_forward_signature

    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])

    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_forward_signature(peft_model)
    ```
    """
    current_signature = inspect.signature(model.forward)
    if len(current_signature.parameters) == 2 and 'args' in current_signature.parameters and ('kwargs' in current_signature.parameters):
        forward = deepcopy(model.forward.__func__)
        update_wrapper(forward, type(model.get_base_model()).forward, assigned=('__doc__', '__name__', '__annotations__'))
        model.forward = MethodType(forward, model)