import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@classmethod
def _convert_dict_to_messages(cls, sa_data: Dict) -> List[BaseMessage]:
    """Convert a dict to a list of BaseMessages."""
    schema = sa_data['schema']
    system = sa_data['system']
    messages = sa_data['messages']
    LOG.info(f'Importing prompt for schema: {schema}')
    result_list: List[BaseMessage] = []
    result_list.append(SystemMessage(content=system))
    result_list.extend([cls._convert_message_from_dict(m) for m in messages])
    return result_list