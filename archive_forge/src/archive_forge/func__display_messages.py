from io import StringIO
from typing import Any, Callable, Dict, List, Mapping, Optional
import yaml
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
def _display_messages(messages: List[BaseMessage]) -> None:
    dict_messages = messages_to_dict(messages)
    for message in dict_messages:
        yaml_string = yaml.dump(message, default_flow_style=False, sort_keys=False, allow_unicode=True, width=10000, line_break=None)
        print('\n', '======= start of message =======', '\n\n')
        print(yaml_string)
        print('======= end of message =======', '\n\n')