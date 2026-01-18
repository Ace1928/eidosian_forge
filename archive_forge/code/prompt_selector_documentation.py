from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
Get default prompt for a language model.

        Args:
            llm: Language model to get prompt for.

        Returns:
            Prompt to use for the language model.
        