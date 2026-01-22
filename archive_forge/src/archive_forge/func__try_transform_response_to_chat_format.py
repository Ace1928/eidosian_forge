from __future__ import annotations
import json
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Union
import langchain.chains
import pydantic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AIMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
@staticmethod
def _try_transform_response_to_chat_format(response):
    if isinstance(response, str):
        message_content = response
        message_id = None
    elif isinstance(response, AIMessage):
        message_content = response.content
        message_id = getattr(response, 'id', None)
    else:
        return response
    transformed_response = _ChatResponse(id=message_id, created=int(time.time()), model=None, choices=[_ChatChoice(index=0, message=_ChatMessage(role='assistant', content=message_content), finish_reason=None)], usage=_ChatUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))
    if Version(pydantic.__version__) < Version('2.0'):
        return json.loads(transformed_response.json())
    else:
        return transformed_response.model_dump(mode='json')