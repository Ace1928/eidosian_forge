import logging
from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional
from langchain_community.utilities.redis import get_client
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
from langchain.memory.utils import get_prompt_input_key
def _create_table_if_not_exists(self) -> None:
    create_table_query = f'\n            CREATE TABLE IF NOT EXISTS {self.full_table_name} (\n                key TEXT PRIMARY KEY,\n                value TEXT\n            )\n        '
    with self.conn:
        self.conn.execute(create_table_query)