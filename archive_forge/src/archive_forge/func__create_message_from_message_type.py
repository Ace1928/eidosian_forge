from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain_core.messages.ai import (
from langchain_core.messages.base import (
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
def _create_message_from_message_type(message_type: str, content: str, name: Optional[str]=None, tool_call_id: Optional[str]=None, **additional_kwargs: Any) -> BaseMessage:
    """Create a message from a message type and content string.

    Args:
        message_type: str the type of the message (e.g., "human", "ai", etc.)
        content: str the content string.

    Returns:
        a message of the appropriate type.
    """
    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs['name'] = name
    if tool_call_id is not None:
        kwargs['tool_call_id'] = tool_call_id
    if additional_kwargs:
        kwargs['additional_kwargs'] = additional_kwargs
    if message_type in ('human', 'user'):
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in ('ai', 'assistant'):
        message = AIMessage(content=content, **kwargs)
    elif message_type == 'system':
        message = SystemMessage(content=content, **kwargs)
    elif message_type == 'function':
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == 'tool':
        message = ToolMessage(content=content, **kwargs)
    else:
        raise ValueError(f"Unexpected message type: {message_type}. Use one of 'human', 'user', 'ai', 'assistant', or 'system'.")
    return message