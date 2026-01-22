import json
from typing import AsyncIterable
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIAPIType, OpenAIConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params
def _prepare_completion_response_payload(self, resp):
    return completions.ResponsePayload(id=resp['id'], object='text_completion', created=resp['created'], model=resp['model'], choices=[completions.Choice(index=idx, text=c['message']['content'], finish_reason=c['finish_reason']) for idx, c in enumerate(resp['choices'])], usage=completions.CompletionsUsage(prompt_tokens=resp['usage']['prompt_tokens'], completion_tokens=resp['usage']['completion_tokens'], total_tokens=resp['usage']['total_tokens']))