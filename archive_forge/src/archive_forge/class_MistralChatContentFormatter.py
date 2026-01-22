import json
import warnings
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.azureml_endpoint import (
class MistralChatContentFormatter(LlamaChatContentFormatter):
    """Content formatter for `Mistral`."""

    def format_messages_request_payload(self, messages: List[BaseMessage], model_kwargs: Dict, api_type: AzureMLEndpointApiType) -> bytes:
        """Formats the request according to the chosen api"""
        chat_messages = [self._convert_message_to_dict(message) for message in messages]
        if chat_messages and chat_messages[0]['role'] == 'system':
            chat_messages[1]['content'] = chat_messages[0]['content'] + '\n\n' + chat_messages[1]['content']
            del chat_messages[0]
        if api_type == AzureMLEndpointApiType.realtime:
            request_payload = json.dumps({'input_data': {'input_string': chat_messages, 'parameters': model_kwargs}})
        elif api_type == AzureMLEndpointApiType.serverless:
            request_payload = json.dumps({'messages': chat_messages, **model_kwargs})
        else:
            raise ValueError(f'`api_type` {api_type} is not supported by this formatter')
        return str.encode(request_payload)