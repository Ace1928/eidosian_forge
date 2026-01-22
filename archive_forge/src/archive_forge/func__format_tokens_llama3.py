import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file
def _format_tokens_llama3(dialogs, tokenizer):
    return [_format_dialog(dialog, tokenizer) for dialog in dialogs]