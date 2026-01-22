from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def _get_output_and_usage_from_tensor(prompt: str, output_tensor: List[int], pipeline, flavor_config, model_config, inference_task):
    """
    Decode the output tensor and return the output text and usage information as a dictionary
    to make the output in OpenAI compatible format.
    """
    usage = _get_token_usage(prompt, output_tensor, pipeline, model_config)
    completions_text = _get_completions_text(prompt, output_tensor, pipeline)
    finish_reason = _get_finish_reason(usage['total_tokens'], usage['completion_tokens'], model_config)
    output_dict = {'id': str(uuid.uuid4()), 'object': _LLM_INFERENCE_OBJECT_NAME[inference_task], 'created': int(time.time()), 'model': flavor_config.get('source_model_name', ''), 'usage': usage}
    completion_choice = {'index': 0, 'finish_reason': finish_reason}
    if inference_task == _LLM_INFERENCE_TASK_COMPLETIONS:
        completion_choice['text'] = completions_text
    elif inference_task == _LLM_INFERENCE_TASK_CHAT:
        completion_choice['message'] = {'role': 'assistant', 'content': completions_text}
    output_dict['choices'] = [completion_choice]
    return output_dict