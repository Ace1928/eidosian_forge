import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, cast
from typing_extensions import TypedDict
from functools import wraps
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run
def _format_result(self, result: Union[EvaluationResult, EvaluationResults, dict], source_run_id: uuid.UUID) -> Union[EvaluationResult, EvaluationResults]:
    if isinstance(result, EvaluationResult):
        if not result.source_run_id:
            result.source_run_id = source_run_id
        return result
    if not isinstance(result, dict):
        raise ValueError(f'Expected a dict, EvaluationResult, or EvaluationResults, got {result}')
    return self._coerce_evaluation_results(result, source_run_id)