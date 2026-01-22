from __future__ import annotations
import inspect
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
def _enter_history(self, input: Any, config: RunnableConfig) -> List[BaseMessage]:
    hist: BaseChatMessageHistory = config['configurable']['message_history']
    messages = hist.messages.copy()
    if not self.history_messages_key:
        input_val = input if not self.input_messages_key else input[self.input_messages_key]
        messages += self._get_input_messages(input_val)
    return messages