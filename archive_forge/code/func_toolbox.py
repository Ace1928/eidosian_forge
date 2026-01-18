import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict
import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate
@property
def toolbox(self) -> Dict[str, Tool]:
    """Get all tool currently available to the agent"""
    return self._toolbox