from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
def _load_rapidfuzz() -> Any:
    """
    Load the RapidFuzz library.

    Raises:
        ImportError: If the rapidfuzz library is not installed.

    Returns:
        Any: The rapidfuzz.distance module.
    """
    try:
        import rapidfuzz
    except ImportError:
        raise ImportError('Please install the rapidfuzz library to use the FuzzyMatchStringEvaluator.Please install it with `pip install rapidfuzz`.')
    return rapidfuzz.distance