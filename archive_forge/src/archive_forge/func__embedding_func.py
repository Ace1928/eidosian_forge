from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.sagemaker_endpoint import ContentHandlerBase
def _embedding_func(self, texts: List[str]) -> List[List[float]]:
    """Call out to SageMaker Inference embedding endpoint."""
    texts = list(map(lambda x: x.replace('\n', ' '), texts))
    _model_kwargs = self.model_kwargs or {}
    _endpoint_kwargs = self.endpoint_kwargs or {}
    body = self.content_handler.transform_input(texts, _model_kwargs)
    content_type = self.content_handler.content_type
    accepts = self.content_handler.accepts
    try:
        response = self.client.invoke_endpoint(EndpointName=self.endpoint_name, Body=body, ContentType=content_type, Accept=accepts, **_endpoint_kwargs)
    except Exception as e:
        raise ValueError(f'Error raised by inference endpoint: {e}')
    return self.content_handler.transform_output(response['Body'])