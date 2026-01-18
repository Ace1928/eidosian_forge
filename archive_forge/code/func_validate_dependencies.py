from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
@root_validator
def validate_dependencies(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """
        Validate that the rapidfuzz library is installed.

        Args:
            values (Dict[str, Any]): The input values.

        Returns:
            Dict[str, Any]: The validated values.
        """
    _load_rapidfuzz()
    return values