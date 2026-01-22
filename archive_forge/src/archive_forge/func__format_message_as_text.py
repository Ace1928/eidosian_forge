import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon
@deprecated('0.0.3', alternative='_convert_messages_to_ollama_messages')
def _format_message_as_text(self, message: BaseMessage) -> str:
    if isinstance(message, ChatMessage):
        message_text = f'\n\n{message.role.capitalize()}: {message.content}'
    elif isinstance(message, HumanMessage):
        if isinstance(message.content, List):
            first_content = cast(List[Dict], message.content)[0]
            content_type = first_content.get('type')
            if content_type == 'text':
                message_text = f'[INST] {first_content['text']} [/INST]'
            elif content_type == 'image_url':
                message_text = first_content['image_url']['url']
        else:
            message_text = f'[INST] {message.content} [/INST]'
    elif isinstance(message, AIMessage):
        message_text = f'{message.content}'
    elif isinstance(message, SystemMessage):
        message_text = f'<<SYS>> {message.content} <</SYS>>'
    else:
        raise ValueError(f'Got unknown type {message}')
    return message_text