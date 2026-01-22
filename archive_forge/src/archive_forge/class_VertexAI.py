from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
@deprecated(since='0.0.12', removal='0.2.0', alternative_import='langchain_google_vertexai.VertexAI')
class VertexAI(_VertexAICommon, BaseLLM):
    """Google Vertex AI large language models."""
    model_name: str = 'text-bison'
    'The name of the Vertex AI large language model.'
    tuned_model_name: Optional[str] = None
    'The name of a tuned model. If provided, model_name is ignored.'

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'llms', 'vertexai']

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        tuned_model_name = values.get('tuned_model_name')
        model_name = values['model_name']
        is_gemini = is_gemini_model(values['model_name'])
        cls._try_init_vertexai(values)
        try:
            from vertexai.language_models import CodeGenerationModel, TextGenerationModel
            from vertexai.preview.language_models import CodeGenerationModel as PreviewCodeGenerationModel
            from vertexai.preview.language_models import TextGenerationModel as PreviewTextGenerationModel
            if is_gemini:
                from vertexai.preview.generative_models import GenerativeModel
            if is_codey_model(model_name):
                model_cls = CodeGenerationModel
                preview_model_cls = PreviewCodeGenerationModel
            elif is_gemini:
                model_cls = GenerativeModel
                preview_model_cls = GenerativeModel
            else:
                model_cls = TextGenerationModel
                preview_model_cls = PreviewTextGenerationModel
            if tuned_model_name:
                values['client'] = model_cls.get_tuned_model(tuned_model_name)
                values['client_preview'] = preview_model_cls.get_tuned_model(tuned_model_name)
            elif is_gemini:
                values['client'] = model_cls(model_name=model_name)
                values['client_preview'] = preview_model_cls(model_name=model_name)
            else:
                values['client'] = model_cls.from_pretrained(model_name)
                values['client_preview'] = preview_model_cls.from_pretrained(model_name)
        except ImportError:
            raise_vertex_import_error()
        if values['streaming'] and values['n'] > 1:
            raise ValueError('Only one candidate can be generated with streaming!')
        return values

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        try:
            result = self.client_preview.count_tokens([text])
        except AttributeError:
            raise_vertex_import_error()
        return result.total_tokens

    def _response_to_generation(self, response: TextGenerationResponse) -> GenerationChunk:
        """Converts a stream response to a generation chunk."""
        try:
            generation_info = {'is_blocked': response.is_blocked, 'safety_attributes': response.safety_attributes}
        except Exception:
            generation_info = None
        return GenerationChunk(text=response.text, generation_info=generation_info)

    def _generate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        params = self._prepare_params(stop=stop, stream=should_stream, **kwargs)
        generations: List[List[Generation]] = []
        for prompt in prompts:
            if should_stream:
                generation = GenerationChunk(text='')
                for chunk in self._stream(prompt, stop=stop, run_manager=run_manager, **kwargs):
                    generation += chunk
                generations.append([generation])
            else:
                res = completion_with_retry(self, [prompt], stream=should_stream, is_gemini=self._is_gemini_model, run_manager=run_manager, **params)
                generations.append([self._response_to_generation(r) for r in res.candidates])
        return LLMResult(generations=generations)

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        params = self._prepare_params(stop=stop, **kwargs)
        generations = []
        for prompt in prompts:
            res = await acompletion_with_retry(self, prompt, is_gemini=self._is_gemini_model, run_manager=run_manager, **params)
            generations.append([self._response_to_generation(r) for r in res.candidates])
        return LLMResult(generations=generations)

    def _stream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        for stream_resp in completion_with_retry(self, [prompt], stream=True, is_gemini=self._is_gemini_model, run_manager=run_manager, **params):
            chunk = self._response_to_generation(stream_resp)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, verbose=self.verbose)
            yield chunk