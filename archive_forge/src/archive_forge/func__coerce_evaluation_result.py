import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, cast
from typing_extensions import TypedDict
from functools import wraps
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run
def _coerce_evaluation_result(self, result: Union[EvaluationResult, dict], source_run_id: uuid.UUID, allow_no_key: bool=False) -> EvaluationResult:
    if isinstance(result, EvaluationResult):
        if not result.source_run_id:
            result.source_run_id = source_run_id
        return result
    try:
        if 'key' not in result:
            if allow_no_key:
                result['key'] = self._name
        return EvaluationResult(**{'source_run_id': source_run_id, **result})
    except ValidationError as e:
        raise ValueError(f"Expected an EvaluationResult object, or dict with a metric 'key' and optional 'score'; got {result}") from e