import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
def _load_pickled_fn_from_hex_string(data: str, allow_dangerous_deserialization: Optional[bool]) -> Callable:
    """Loads a pickled function from a hexadecimal string."""
    if not allow_dangerous_deserialization:
        raise ValueError('This code relies on the pickle module. You will need to set allow_dangerous_deserialization=True if you want to opt-in to allow deserialization of data using pickle.Data can be compromised by a malicious actor if not handled properly to include a malicious payload that when deserialized with pickle can execute arbitrary code on your machine.')
    try:
        import cloudpickle
    except Exception as e:
        raise ValueError(f'Please install cloudpickle>=2.0.0. Error: {e}')
    try:
        return cloudpickle.loads(bytes.fromhex(data))
    except Exception as e:
        raise ValueError(f'Failed to load the pickled function from a hexadecimal string. Error: {e}')