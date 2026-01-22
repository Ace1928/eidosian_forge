from typing import Any, Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.amadeus.base import AmadeusBaseTool
Tool's llm used for calculating the closest airport. Defaults to `ChatOpenAI`.