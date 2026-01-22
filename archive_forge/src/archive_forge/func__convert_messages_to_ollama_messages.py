import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon
def _convert_messages_to_ollama_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Union[str, List[str]]]]:
    ollama_messages: List = []
    for message in messages:
        role = ''
        if isinstance(message, HumanMessage):
            role = 'user'
        elif isinstance(message, AIMessage):
            role = 'assistant'
        elif isinstance(message, SystemMessage):
            role = 'system'
        else:
            raise ValueError('Received unsupported message type for Ollama.')
        content = ''
        images = []
        if isinstance(message.content, str):
            content = message.content
        else:
            for content_part in cast(List[Dict], message.content):
                if content_part.get('type') == 'text':
                    content += f'\n{content_part['text']}'
                elif content_part.get('type') == 'image_url':
                    if isinstance(content_part.get('image_url'), str):
                        image_url_components = content_part['image_url'].split(',')
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])
                    else:
                        raise ValueError('Only string image_url content parts are supported.')
                else:
                    raise ValueError("Unsupported message content type. Must either have type 'text' or type 'image_url' with a string 'image_url' field.")
        ollama_messages.append({'role': role, 'content': content, 'images': images})
    return ollama_messages