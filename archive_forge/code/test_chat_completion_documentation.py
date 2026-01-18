import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file

        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        