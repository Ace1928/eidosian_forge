import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _token_count_eval_fn(predictions, targets=None, metrics=None):
    import tiktoken
    os.environ['TIKTOKEN_CACHE_DIR'] = ''
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = []
    for prediction in predictions:
        if isinstance(prediction, str):
            num_tokens.append(len(encoding.encode(prediction)))
        else:
            num_tokens.append(None)
    return MetricValue(scores=num_tokens, aggregate_results={})