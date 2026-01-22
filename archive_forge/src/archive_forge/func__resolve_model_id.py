from typing import Any, AsyncIterator, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.huggingface_text_gen_inference import (
def _resolve_model_id(self) -> None:
    """Resolve the model_id from the LLM's inference_server_url"""
    from huggingface_hub import list_inference_endpoints
    available_endpoints = list_inference_endpoints('*')
    if isinstance(self.llm, HuggingFaceHub) or (hasattr(self.llm, 'repo_id') and self.llm.repo_id):
        self.model_id = self.llm.repo_id
        return
    elif isinstance(self.llm, HuggingFaceTextGenInference):
        endpoint_url: Optional[str] = self.llm.inference_server_url
    else:
        endpoint_url = self.llm.endpoint_url
    for endpoint in available_endpoints:
        if endpoint.url == endpoint_url:
            self.model_id = endpoint.repository
    if not self.model_id:
        raise ValueError(f'Failed to resolve model_id:Could not find model id for inference server: {endpoint_url}Make sure that your Hugging Face token has access to the endpoint.')