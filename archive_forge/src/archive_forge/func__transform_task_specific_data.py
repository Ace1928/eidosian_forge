import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pytz
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
@staticmethod
def _transform_task_specific_data(task: str, input_data: Union[List[Any], Any], response: Union[List[Any], Any]) -> Tuple[Union[List[Any], Any], Union[List[Any], Any]]:
    """Transform input and response data based on specific tasks.

        :param task: the task name
        :param input_data: the input data
        :param response: the response data
        :returns: tuple of transformed input_data and response
        """
    if task == 'question-answering':
        input_data = input_data if isinstance(input_data, list) else [input_data]
        input_data = [data.__dict__ for data in input_data]
    elif task == 'conversational':
        input_data = input_data if isinstance(input_data, list) else [input_data]
        input_data = [data.__dict__['past_user_inputs'][-1] for data in input_data]
        response = response if isinstance(response, list) else [response]
        response = [data.__dict__['generated_responses'][-1] for data in response]
    return (input_data, response)