import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, cast
from typing_extensions import TypedDict
from functools import wraps
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run
class FeedbackConfig(TypedDict, total=False):
    """Configuration to define a type of feedback.

    Applied on on the first creation of a feedback_key.
    """
    type: Literal['continuous', 'categorical', 'freeform']
    'The type of feedback.'
    min: Optional[Union[float, int]]
    'The minimum permitted value (if continuous type).'
    max: Optional[Union[float, int]]
    'The maximum value permitted value (if continuous type).'
    categories: Optional[List[Union[Category, dict]]]