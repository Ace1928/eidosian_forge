from __future__ import annotations
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_core.utils.utils import convert_to_secret_str
class Friendli(LLM, BaseFriendli):
    """Friendli LLM.

    ``friendli-client`` package should be installed with `pip install friendli-client`.
    You must set ``FRIENDLI_TOKEN`` environment variable or provide the value of your
    personal access token for the ``friendli_token`` argument.

    Example:
        .. code-block:: python

            from langchain_community.llms import Friendli

            friendli = Friendli(
                model="mixtral-8x7b-instruct-v0-1", friendli_token="YOUR FRIENDLI TOKEN"
            )
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'friendli_token': 'FRIENDLI_TOKEN'}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Friendli completions API."""
        return {'frequency_penalty': self.frequency_penalty, 'presence_penalty': self.presence_penalty, 'max_tokens': self.max_tokens, 'stop': self.stop, 'temperature': self.temperature, 'top_p': self.top_p}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'model': self.model, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'friendli'

    def _get_invocation_params(self, stop: Optional[List[str]]=None, **kwargs: Any) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params = self._default_params
        if self.stop is not None and stop is not None:
            raise ValueError('`stop` found in both the input and default params.')
        elif self.stop is not None:
            params['stop'] = self.stop
        else:
            params['stop'] = stop
        return {**params, **kwargs}

    def _call(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Call out Friendli's completions API.

        Args:
            prompt (str): The text prompt to generate completion for.
            stop (Optional[List[str]], optional): When one of the stop phrases appears
                in the generation result, the API will stop generation. The stop phrases
                are excluded from the result. If beam search is enabled, all of the
                active beams should contain the stop phrase to terminate generation.
                Before checking whether a stop phrase is included in the result, the
                phrase is converted into tokens. We recommend using stop_tokens because
                it is clearer. For example, after tokenization, phrases "clear" and
                " clear" can result in different token sequences due to the prepended
                space character. Defaults to None.

        Returns:
            str: The generated text output.

        Example:
            .. code-block:: python

                response = frienldi("Give me a recipe for the Old Fashioned cocktail.")
        """
        params = self._get_invocation_params(stop=stop, **kwargs)
        completion = self.client.completions.create(model=self.model, prompt=prompt, stream=False, **params)
        return completion.choices[0].text

    async def _acall(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Call out Friendli's completions API Asynchronously.

        Args:
            prompt (str): The text prompt to generate completion for.
            stop (Optional[List[str]], optional): When one of the stop phrases appears
                in the generation result, the API will stop generation. The stop phrases
                are excluded from the result. If beam search is enabled, all of the
                active beams should contain the stop phrase to terminate generation.
                Before checking whether a stop phrase is included in the result, the
                phrase is converted into tokens. We recommend using stop_tokens because
                it is clearer. For example, after tokenization, phrases "clear" and
                " clear" can result in different token sequences due to the prepended
                space character. Defaults to None.

        Returns:
            str: The generated text output.

        Example:
            .. code-block:: python

                response = await frienldi("Tell me a joke.")
        """
        params = self._get_invocation_params(stop=stop, **kwargs)
        completion = await self.async_client.completions.create(model=self.model, prompt=prompt, stream=False, **params)
        return completion.choices[0].text

    def _stream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
        params = self._get_invocation_params(stop=stop, **kwargs)
        stream = self.client.completions.create(model=self.model, prompt=prompt, stream=True, **params)
        for line in stream:
            chunk = _stream_response_to_generation_chunk(line)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(line.text, chunk=chunk)

    async def _astream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        params = self._get_invocation_params(stop=stop, **kwargs)
        stream = await self.async_client.completions.create(model=self.model, prompt=prompt, stream=True, **params)
        async for line in stream:
            chunk = _stream_response_to_generation_chunk(line)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(line.text, chunk=chunk)

    def _generate(self, prompts: list[str], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        """Call out Friendli's completions API with k unique prompts.

        Args:
            prompt (str): The text prompt to generate completion for.
            stop (Optional[List[str]], optional): When one of the stop phrases appears
                in the generation result, the API will stop generation. The stop phrases
                are excluded from the result. If beam search is enabled, all of the
                active beams should contain the stop phrase to terminate generation.
                Before checking whether a stop phrase is included in the result, the
                phrase is converted into tokens. We recommend using stop_tokens because
                it is clearer. For example, after tokenization, phrases "clear" and
                " clear" can result in different token sequences due to the prepended
                space character. Defaults to None.

        Returns:
            str: The generated text output.

        Example:
            .. code-block:: python

                response = frienldi.generate(["Tell me a joke."])
        """
        llm_output = {'model': self.model}
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError('Cannot stream results with multiple prompts.')
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]], llm_output=llm_output)
        llm_result = super()._generate(prompts, stop, run_manager, **kwargs)
        llm_result.llm_output = llm_output
        return llm_result

    async def _agenerate(self, prompts: list[str], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        """Call out Friendli's completions API asynchronously with k unique prompts.

        Args:
            prompt (str): The text prompt to generate completion for.
            stop (Optional[List[str]], optional): When one of the stop phrases appears
                in the generation result, the API will stop generation. The stop phrases
                are excluded from the result. If beam search is enabled, all of the
                active beams should contain the stop phrase to terminate generation.
                Before checking whether a stop phrase is included in the result, the
                phrase is converted into tokens. We recommend using stop_tokens because
                it is clearer. For example, after tokenization, phrases "clear" and
                " clear" can result in different token sequences due to the prepended
                space character. Defaults to None.

        Returns:
            str: The generated text output.

        Example:
            .. code-block:: python

                response = await frienldi.agenerate(
                    ["Give me a recipe for the Old Fashioned cocktail."]
                )
        """
        llm_output = {'model': self.model}
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError('Cannot stream results with multiple prompts.')
            generation = None
            async for chunk in self._astream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]], llm_output=llm_output)
        llm_result = await super()._agenerate(prompts, stop, run_manager, **kwargs)
        llm_result.llm_output = llm_output
        return llm_result