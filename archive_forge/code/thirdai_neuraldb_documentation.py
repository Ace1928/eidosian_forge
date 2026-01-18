from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
Saves a NeuralDB instance to disk. Can be loaded into memory by
        calling NeuralDB.from_checkpoint(path)

        Args:
            path: path on disk to save the NeuralDB instance to.
        