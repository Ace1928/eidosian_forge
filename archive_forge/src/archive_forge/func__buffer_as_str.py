from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.pydantic_v1 import root_validator
from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key
def _buffer_as_str(self, messages: List[BaseMessage]) -> str:
    return get_buffer_string(messages, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)