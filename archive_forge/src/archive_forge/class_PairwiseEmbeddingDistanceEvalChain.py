from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
from langchain.utils.math import cosine_similarity
class PairwiseEmbeddingDistanceEvalChain(_EmbeddingDistanceChainMixin, PairwiseStringEvaluator):
    """Use embedding distances to score semantic difference between two predictions.

    Examples:
    >>> chain = PairwiseEmbeddingDistanceEvalChain()
    >>> result = chain.evaluate_string_pairs(prediction="Hello", prediction_b="Hi")
    >>> print(result)
    {'score': 0.5}
    """

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys of the chain.

        Returns:
            List[str]: The input keys.
        """
        return ['prediction', 'prediction_b']

    @property
    def evaluation_name(self) -> str:
        return f'pairwise_embedding_{self.distance_metric.value}_distance'

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Compute the score for two predictions.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (CallbackManagerForChainRun, optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        vectors = np.array(self.embeddings.embed_documents([inputs['prediction'], inputs['prediction_b']]))
        score = self._compute_score(vectors)
        return {'score': score}

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Asynchronously compute the score for two predictions.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (AsyncCallbackManagerForChainRun, optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        embedded = await self.embeddings.aembed_documents([inputs['prediction'], inputs['prediction_b']])
        vectors = np.array(embedded)
        score = self._compute_score(vectors)
        return {'score': score}

    def _evaluate_string_pairs(self, *, prediction: str, prediction_b: str, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Evaluate the embedding distance between two predictions.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces
            metadata (Dict[str, Any], optional): metadata to apply to
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
        """
        result = self(inputs={'prediction': prediction, 'prediction_b': prediction_b}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)

    async def _aevaluate_string_pairs(self, *, prediction: str, prediction_b: str, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Asynchronously evaluate the embedding distance

        between two predictions.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces
            metadata (Dict[str, Any], optional): metadata to apply to traces
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
        """
        result = await self.acall(inputs={'prediction': prediction, 'prediction_b': prediction_b}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)