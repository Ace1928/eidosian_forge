from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
def _prepare_invocation_object(self, prompt: str, stop: Optional[List[str]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from oci.generative_ai_inference import models
    oci_llm_request_mapping = {'cohere': models.CohereLlmInferenceRequest, 'meta': models.LlamaLlmInferenceRequest}
    provider = self._get_provider()
    _model_kwargs = self.model_kwargs or {}
    if stop is not None:
        _model_kwargs[self.llm_stop_sequence_mapping[provider]] = stop
    if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
        serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
    else:
        serving_mode = models.OnDemandServingMode(model_id=self.model_id)
    inference_params = {**_model_kwargs, **kwargs}
    inference_params['prompt'] = prompt
    inference_params['is_stream'] = self.is_stream
    invocation_obj = models.GenerateTextDetails(compartment_id=self.compartment_id, serving_mode=serving_mode, inference_request=oci_llm_request_mapping[provider](**inference_params))
    return invocation_obj