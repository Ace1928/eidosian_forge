from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
class StringDistanceEvalChain(StringEvaluator, _RapidFuzzChainMixin):
    """Compute string distances between the prediction and the reference.

    Examples
    ----------

    >>> from langchain.evaluation import StringDistanceEvalChain
    >>> evaluator = StringDistanceEvalChain()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CEO",
        )

    Using the `load_evaluator` function:

    >>> from langchain.evaluation import load_evaluator
    >>> evaluator = load_evaluator("string_distance")
    >>> evaluator.evaluate_strings(
            prediction="The answer is three",
            reference="three",
        )
    """

    @property
    def requires_input(self) -> bool:
        """
        This evaluator does not require input.
        """
        return False

    @property
    def requires_reference(self) -> bool:
        """
        This evaluator does not require a reference.
        """
        return True

    @property
    def input_keys(self) -> List[str]:
        """
        Get the input keys.

        Returns:
            List[str]: The input keys.
        """
        return ['reference', 'prediction']

    @property
    def evaluation_name(self) -> str:
        """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return f'{self.distance.value}_distance'

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """
        Compute the string distance between the prediction and the reference.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (Optional[CallbackManagerForChainRun]):
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {'score': self.compute_metric(inputs['reference'], inputs['prediction'])}

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """
        Asynchronously compute the string distance between the prediction
            and the reference.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (Optional[AsyncCallbackManagerForChainRun]:
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {'score': self.compute_metric(inputs['reference'], inputs['prediction'])}

    def _evaluate_strings(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """
        Evaluate the string distance between the prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference string.
            input (Optional[str], optional): The input string.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
        result = self(inputs={'prediction': prediction, 'reference': reference}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)

    async def _aevaluate_strings(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """
        Asynchronously evaluate the string distance between the
            prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference string.
            input (Optional[str], optional): The input string.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
        result = await self.acall(inputs={'prediction': prediction, 'reference': reference}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)