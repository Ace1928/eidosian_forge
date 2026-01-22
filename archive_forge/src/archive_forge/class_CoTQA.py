from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run
from langchain.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain.evaluation.embedding_distance.base import (
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.evaluation.string_distance.base import (
class CoTQA(SingleKeyEvalConfig):
    """Configuration for a context-based QA evaluator.

        Parameters
        ----------
        prompt : Optional[BasePromptTemplate]
            The prompt template to use for generating the question.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.

        """
    evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
    llm: Optional[BaseLanguageModel] = None
    prompt: Optional[BasePromptTemplate] = None