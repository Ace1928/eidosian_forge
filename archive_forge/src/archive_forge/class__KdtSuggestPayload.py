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
class _KdtSuggestPayload(BaseModel):
    """pydantic API request type"""
    question: Optional[str]
    context: List[_KdtSuggestContext]

    def get_system_str(self) -> str:
        lines = []
        for table_context in self.context:
            if table_context.table is None:
                continue
            context_str = table_context.to_system_str()
            lines.append(context_str)
        return '\n\n'.join(lines)

    def get_messages(self) -> List[Dict]:
        messages = []
        for context in self.context:
            if context.samples is None:
                continue
            for question, answer in context.samples.items():
                answer = answer.replace("''", "'")
                messages.append(dict(role='user', content=question or ''))
                messages.append(dict(role='assistant', content=answer))
        return messages

    def to_completion(self) -> Dict:
        messages = []
        messages.append(dict(role='system', content=self.get_system_str()))
        messages.extend(self.get_messages())
        messages.append(dict(role='user', content=self.question or ''))
        response = dict(messages=messages)
        return response