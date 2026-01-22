from typing import List, Optional
from mlflow.exceptions import MlflowException
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import make_genai_metric
from mlflow.metrics.genai.utils import _get_latest_metric_version
from mlflow.models import EvaluationMetric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
@experimental
def answer_relevance(model: Optional[str]=None, metric_version: Optional[str]=_get_latest_metric_version(), examples: Optional[List[EvaluationExample]]=None) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the answer relevance of an LLM
    using the model provided. Answer relevance will be assessed based on the appropriateness and
    applicability of the output with respect to the input.

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: Model uri of an openai or gateway judge model in the format of
            "openai:/gpt-4" or "gateway:/my-route". Defaults to
            "openai:/gpt-4". Your use of a third party LLM service (e.g., OpenAI) for
            evaluation may be subject to and governed by the LLM service's terms of use.
        metric_version: The version of the answer relevance metric to use.
            Defaults to the latest version.
        examples: Provide a list of examples to help the judge model evaluate the
            answer relevance. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.

    Returns:
        A metric object
    """
    class_name = f'mlflow.metrics.genai.prompts.{metric_version}.AnswerRelevanceMetric'
    try:
        answer_relevance_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(f'Failed to find answer relevance metric for version {metric_version}. Please check the version', error_code=INVALID_PARAMETER_VALUE) from None
    except Exception as e:
        raise MlflowException(f'Failed to construct answer relevance metric {metric_version}. Error: {e!r}', error_code=INTERNAL_ERROR) from None
    if examples is None:
        examples = answer_relevance_class_module.default_examples
    if model is None:
        model = answer_relevance_class_module.default_model
    return make_genai_metric(name='answer_relevance', definition=answer_relevance_class_module.definition, grading_prompt=answer_relevance_class_module.grading_prompt, examples=examples, version=metric_version, model=model, parameters=answer_relevance_class_module.parameters, aggregations=['mean', 'variance', 'p90'], greater_is_better=True)