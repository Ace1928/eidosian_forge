from io import StringIO
from typing import Any, Callable, Dict, List, Mapping, Optional
import yaml
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
def _collect_yaml_input(messages: List[BaseMessage], stop: Optional[List[str]]=None) -> BaseMessage:
    """Collects and returns user input as a single string."""
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        if stop and any((seq in line for seq in stop)):
            break
        lines.append(line)
    yaml_string = '\n'.join(lines)
    try:
        message = _message_from_dict(yaml.safe_load(StringIO(yaml_string)))
        if message is None:
            return HumanMessage(content='')
        if stop:
            if isinstance(message.content, str):
                message.content = enforce_stop_tokens(message.content, stop)
            else:
                raise ValueError('Cannot use when output is not a string.')
        return message
    except yaml.YAMLError:
        raise ValueError('Invalid YAML string entered.')
    except ValueError:
        raise ValueError('Invalid message entered.')