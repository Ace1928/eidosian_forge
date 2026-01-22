from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.comparison.prompt import (
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.schema import LLMEvalChain, PairwiseStringEvaluator
from langchain.schema import RUN_KEY
class PairwiseStringEvalChain(PairwiseStringEvaluator, LLMEvalChain, LLMChain):
    """A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    Example:
        >>> from langchain_community.chat_models import ChatOpenAI
        >>> from langchain.evaluation.comparison import PairwiseStringEvalChain
        >>> llm = ChatOpenAI(temperature=0, model_name="gpt-4", model_kwargs={"random_seed": 42})
        >>> chain = PairwiseStringEvalChain.from_llm(llm=llm)
        >>> result = chain.evaluate_string_pairs(
        ...     input = "What is the chemical formula for water?",
        ...     prediction = "H2O",
        ...     prediction_b = (
        ...        "The chemical formula for water is H2O, which means"
        ...        " there are two hydrogen atoms and one oxygen atom."
        ...     reference = "The chemical formula for water is H2O.",
        ... )
        >>> print(result)
        # {
        #    "value": "B",
        #    "comment": "Both responses accurately state"
        #       " that the chemical formula for water is H2O."
        #       " However, Response B provides additional information"
        # .     " by explaining what the formula means.\\n[[B]]"
        # }

    """
    output_key: str = 'results'
    output_parser: BaseOutputParser = Field(default_factory=PairwiseStringResultOutputParser)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    class Config:
        """Configuration for the PairwiseStringEvalChain."""
        extra = Extra.ignore

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return False

    @property
    def requires_input(self) -> bool:
        """Return whether the chain requires an input.

        Returns:
            bool: True if the chain requires an input, False otherwise.

        """
        return True

    @property
    def _skip_reference_warning(self) -> str:
        """Return the warning to show when reference is ignored.

        Returns:
            str: The warning to show when reference is ignored.

        """
        return f'Ignoring reference in {self.__class__.__name__}, as it is not expected.\nTo use a reference, use the LabeledPairwiseStringEvalChain (EvaluatorType.LABELED_PAIRWISE_STRING) instead.'

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, *, prompt: Optional[PromptTemplate]=None, criteria: Optional[Union[CRITERIA_TYPE, str]]=None, **kwargs: Any) -> PairwiseStringEvalChain:
        """Initialize the PairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseChatModel): The LLM to use (GPT-4 recommended).
            prompt (PromptTemplate, optional): The prompt to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            PairwiseStringEvalChain: The initialized PairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        if not (isinstance(llm, (ChatOpenAI, AzureChatOpenAI)) and llm.model_name.startswith('gpt-4')):
            logger.warning('This chain was only tested with GPT-4. Performance may be significantly worse with other models.')
        expected_input_vars = {'prediction', 'prediction_b', 'input', 'criteria'}
        prompt_ = prompt or COMPARISON_TEMPLATE.partial(reference='')
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt_.input_variables}')
        criteria_ = resolve_pairwise_criteria(criteria)
        criteria_str = '\n'.join((f'{k}: {v}' if v else k for k, v in criteria_.items()))
        criteria_str = CRITERIA_INSTRUCTIONS + criteria_str if criteria_str else ''
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), **kwargs)

    def _prepare_input(self, prediction: str, prediction_b: str, input: Optional[str], reference: Optional[str]) -> dict:
        """Prepare the input for the chain.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            reference (str, optional): The reference string, if any.

        Returns:
            dict: The prepared input for the chain.

        """
        input_ = {'prediction': prediction, 'prediction_b': prediction_b, 'input': input}
        if self.requires_reference:
            input_['reference'] = reference
        return input_

    def _prepare_output(self, result: dict) -> dict:
        """Prepare the output."""
        parsed = result[self.output_key]
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        return parsed

    def _evaluate_string_pairs(self, *, prediction: str, prediction_b: str, input: Optional[str]=None, reference: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Evaluate whether output A is preferred to output B.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = self(inputs=input_, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)

    async def _aevaluate_string_pairs(self, *, prediction: str, prediction_b: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Asynchronously evaluate whether output A is preferred to output B.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = await self.acall(inputs=input_, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)