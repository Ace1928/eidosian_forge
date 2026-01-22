from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def _get_token_usage(prompt: str, output_tensor: List[int], pipeline, model_config):
    """Return the prompt tokens, completion tokens, and the total tokens as dict."""
    inputs = pipeline.tokenizer(prompt, return_tensors=pipeline.framework, max_length=model_config.get('max_length', None), add_special_tokens=False)
    prompt_tokens = inputs['input_ids'].shape[-1]
    total_tokens = len(output_tensor)
    completions_tokens = total_tokens - prompt_tokens
    return {'prompt_tokens': prompt_tokens, 'completion_tokens': completions_tokens, 'total_tokens': total_tokens}