from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def token_count() -> EvaluationMetric:
    """
    This function will create a metric for calculating token_count. Token count is calculated
    using tiktoken by using the `cl100k_base` tokenizer.
    """
    return make_metric(eval_fn=_token_count_eval_fn, greater_is_better=True, name='token_count')