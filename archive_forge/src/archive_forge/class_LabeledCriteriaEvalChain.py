from __future__ import annotations
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.criteria.prompt import PROMPT, PROMPT_WITH_REFERENCES
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.schema import RUN_KEY
class LabeledCriteriaEvalChain(CriteriaEvalChain):
    """Criteria evaluation chain that requires references."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        """Whether the evaluation requires a reference text."""
        return True

    @classmethod
    def _resolve_prompt(cls, prompt: Optional[BasePromptTemplate]=None) -> BasePromptTemplate:
        expected_input_vars = {'input', 'output', 'criteria', 'reference'}
        prompt_ = prompt or PROMPT_WITH_REFERENCES
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt_.input_variables}')
        return prompt_

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, criteria: Optional[CRITERIA_TYPE]=None, *, prompt: Optional[BasePromptTemplate]=None, **kwargs: Any) -> CriteriaEvalChain:
        """Create a `LabeledCriteriaEvalChain` instance from an llm and criteria.

        Parameters
        ----------
        llm : BaseLanguageModel
            The language model to use for evaluation.
        criteria : CRITERIA_TYPE - default=None for "helpfulness"
            The criteria to evaluate the runs against. It can be:
                -  a mapping of a criterion name to its description
                -  a single criterion name present in one of the default criteria
                -  a single `ConstitutionalPrinciple` instance
        prompt : Optional[BasePromptTemplate], default=None
            The prompt template to use for generating prompts. If not provided,
            a default prompt will be used.
        **kwargs : Any
            Additional keyword arguments to pass to the `LLMChain`
            constructor.

        Returns
        -------
        LabeledCriteriaEvalChain
            An instance of the `LabeledCriteriaEvalChain` class.

        Examples
        --------
        >>> from langchain_community.llms import OpenAI
        >>> from langchain.evaluation.criteria import LabeledCriteriaEvalChain
        >>> llm = OpenAI()
        >>> criteria = {
                "hallucination": (
                    "Does this submission contain information"
                    " not present in the input or reference?"
                ),
            }
        >>> chain = LabeledCriteriaEvalChain.from_llm(
                llm=llm,
                criteria=criteria,
            )
        """
        prompt = cls._resolve_prompt(prompt)
        criteria_ = cls.resolve_criteria(criteria)
        criteria_str = '\n'.join((f'{k}: {v}' for k, v in criteria_.items()))
        prompt_ = prompt.partial(criteria=criteria_str)
        return cls(llm=llm, prompt=prompt_, criterion_name='-'.join(criteria_), **kwargs)