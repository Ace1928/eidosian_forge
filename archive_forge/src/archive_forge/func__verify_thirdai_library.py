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
@staticmethod
def _verify_thirdai_library(thirdai_key: Optional[str]=None) -> None:
    try:
        from thirdai import licensing
        importlib.util.find_spec('thirdai.neural_db')
        licensing.activate(thirdai_key or os.getenv('THIRDAI_KEY'))
    except ImportError:
        raise ModuleNotFoundError('Could not import thirdai python package and neuraldb dependencies. Please install it with `pip install thirdai[neural_db]`.')