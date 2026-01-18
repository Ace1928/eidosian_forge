from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def toxicity() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `toxicity`_ using the model
    `roberta-hate-speech-dynabench-r4`_, which defines hate as "abusive speech targeting
    specific group characteristics, such as ethnic origin, religion, gender, or sexual
    orientation."

    The score ranges from 0 to 1, where scores closer to 1 are more toxic. The default threshold
    for a text to be considered "toxic" is 0.5.

    Aggregations calculated for this metric:
        - ratio (of toxic input texts)

    .. _toxicity: https://huggingface.co/spaces/evaluate-measurement/toxicity
    .. _roberta-hate-speech-dynabench-r4: https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target
    """
    return make_metric(eval_fn=_toxicity_eval_fn, greater_is_better=False, name='toxicity', long_name='toxicity/roberta-hate-speech-dynabench-r4', version='v1')