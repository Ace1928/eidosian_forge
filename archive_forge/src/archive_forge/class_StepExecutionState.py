import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml
from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime
class StepExecutionState:
    """
    Represents execution state for a step, including the current status and
    the time of the last status update.
    """
    _KEY_STATUS = 'recipe_step_execution_status'
    _KEY_LAST_UPDATED_TIMESTAMP = 'recipe_step_execution_last_updated_timestamp'
    _KEY_STACK_TRACE = 'recipe_step_stack_trace'

    def __init__(self, status: StepStatus, last_updated_timestamp: int, stack_trace: str):
        """
        Args:
            status: The execution status of the step.
            last_updated_timestamp: The timestamp of the last execution status update, measured
                in seconds since the UNIX epoch.
            stack_trace: The stack trace of the last execution. None if the step execution
                succeeds.
        """
        self.status = status
        self.last_updated_timestamp = last_updated_timestamp
        self.stack_trace = stack_trace

    def to_dict(self) -> Dict[str, Any]:
        """
        Creates a dictionary representation of the step execution state.
        """
        return {StepExecutionState._KEY_STATUS: self.status.value, StepExecutionState._KEY_LAST_UPDATED_TIMESTAMP: self.last_updated_timestamp, StepExecutionState._KEY_STACK_TRACE: self.stack_trace}

    @classmethod
    def from_dict(cls, state_dict) -> 'StepExecutionState':
        """
        Creates a ``StepExecutionState`` instance from the specified execution state dictionary.
        """
        return cls(status=StepStatus[state_dict[StepExecutionState._KEY_STATUS]], last_updated_timestamp=state_dict[StepExecutionState._KEY_LAST_UPDATED_TIMESTAMP], stack_trace=state_dict[StepExecutionState._KEY_STACK_TRACE])