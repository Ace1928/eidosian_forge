import json
import logging
import re
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _sanitize_input(self, input_str: str) -> str:
    return re.sub('[^a-zA-Z0-9_]', '', input_str)