import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, cast
from typing_extensions import TypedDict
from functools import wraps
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run
class EvaluationResults(TypedDict, total=False):
    """Batch evaluation results.

    This makes it easy for your evaluator to return multiple
    metrics at once.
    """
    results: List[EvaluationResult]
    'The evaluation results.'