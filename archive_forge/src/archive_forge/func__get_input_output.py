import warnings
from abc import ABC
from typing import Any, Dict, Optional, Tuple
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Field
from langchain.memory.utils import get_prompt_input_key
def _get_input_output(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> Tuple[str, str]:
    if self.input_key is None:
        prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
    else:
        prompt_input_key = self.input_key
    if self.output_key is None:
        if len(outputs) == 1:
            output_key = list(outputs.keys())[0]
        elif 'output' in outputs:
            output_key = 'output'
            warnings.warn(f"'{self.__class__.__name__}' got multiple output keys: {outputs.keys()}. The default 'output' key is being used. If this is not desired, please manually set 'output_key'.")
        else:
            raise ValueError(f"Got multiple output keys: {outputs.keys()}, cannot determine which to store in memory. Please set the 'output_key' explicitly.")
    else:
        output_key = self.output_key
    return (inputs[prompt_input_key], outputs[output_key])