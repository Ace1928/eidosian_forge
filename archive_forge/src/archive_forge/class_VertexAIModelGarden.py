from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
@deprecated(since='0.0.12', removal='0.2.0', alternative_import='langchain_google_vertexai.VertexAIModelGarden')
class VertexAIModelGarden(_VertexAIBase, BaseLLM):
    """Vertex AI Model Garden large language models."""
    client: 'PredictionServiceClient' = None
    async_client: 'PredictionServiceAsyncClient' = None
    endpoint_id: str
    'A name of an endpoint where the model has been deployed.'
    allowed_model_args: Optional[List[str]] = None
    'Allowed optional args to be passed to the model.'
    prompt_arg: str = 'prompt'
    result_arg: Optional[str] = 'generated_text'
    'Set result_arg to None if output of the model is expected to be a string.'
    "Otherwise, if it's a dict, provided an argument that contains the result."

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient, PredictionServiceClient
        except ImportError:
            raise_vertex_import_error()
        if not values['project']:
            raise ValueError('A GCP project should be provided to run inference on Model Garden!')
        client_options = ClientOptions(api_endpoint=f'{values['location']}-aiplatform.googleapis.com')
        client_info = get_client_info(module='vertex-ai-model-garden')
        values['client'] = PredictionServiceClient(client_options=client_options, client_info=client_info)
        values['async_client'] = PredictionServiceAsyncClient(client_options=client_options, client_info=client_info)
        return values

    @property
    def endpoint_path(self) -> str:
        return self.client.endpoint_path(project=self.project, location=self.location, endpoint=self.endpoint_id)

    @property
    def _llm_type(self) -> str:
        return 'vertexai_model_garden'

    def _prepare_request(self, prompts: List[str], **kwargs: Any) -> List['Value']:
        try:
            from google.protobuf import json_format
            from google.protobuf.struct_pb2 import Value
        except ImportError:
            raise ImportError('protobuf package not found, please install it with `pip install protobuf`')
        instances = []
        for prompt in prompts:
            if self.allowed_model_args:
                instance = {k: v for k, v in kwargs.items() if k in self.allowed_model_args}
            else:
                instance = {}
            instance[self.prompt_arg] = prompt
            instances.append(instance)
        predict_instances = [json_format.ParseDict(instance_dict, Value()) for instance_dict in instances]
        return predict_instances

    def _generate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)
        response = self.client.predict(endpoint=self.endpoint_path, instances=instances)
        return self._parse_response(response)

    def _parse_response(self, predictions: 'Prediction') -> LLMResult:
        generations: List[List[Generation]] = []
        for result in predictions.predictions:
            generations.append([Generation(text=self._parse_prediction(prediction)) for prediction in result])
        return LLMResult(generations=generations)

    def _parse_prediction(self, prediction: Any) -> str:
        if isinstance(prediction, str):
            return prediction
        if self.result_arg:
            try:
                return prediction[self.result_arg]
            except KeyError:
                if isinstance(prediction, str):
                    error_desc = f'Provided non-None `result_arg` (result_arg={self.result_arg}). But got prediction of type {type(prediction)} instead of dict. Most probably, youneed to set `result_arg=None` during VertexAIModelGarden initialization.'
                    raise ValueError(error_desc)
                else:
                    raise ValueError(f'{self.result_arg} key not found in prediction!')
        return prediction

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)
        response = await self.async_client.predict(endpoint=self.endpoint_path, instances=instances)
        return self._parse_response(response)