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
class KineticaSqlResponse(BaseModel):
    """Response containing SQL and the fetched data.

    This object is returned by a chain with ``KineticaSqlOutputParser`` and it contains
    the generated SQL and related Pandas Dataframe fetched from the database.
    """
    sql: str = Field(default=None)
    'The generated SQL.'
    dataframe: Any = Field(default=None)
    'The Pandas dataframe containing the fetched data.'

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True