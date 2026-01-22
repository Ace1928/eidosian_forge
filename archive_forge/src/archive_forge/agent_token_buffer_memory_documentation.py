from typing import Any, Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain.agents.format_scratchpad.openai_functions import (
from langchain.memory.chat_memory import BaseChatMemory
Save context from this conversation to buffer. Pruned.