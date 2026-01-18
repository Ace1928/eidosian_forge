from typing import Any, Dict, List
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.pydantic_v1 import root_validator
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.summary import SummarizerMixin
Clear memory contents.