import logging
from time import perf_counter
from typing import Any, Dict, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.tools import BaseTool
from langchain_community.chat_models.openai import _import_tiktoken
from langchain_community.tools.powerbi.prompt import (
from langchain_community.utilities.powerbi import PowerBIDataset, json_to_md
def _result_too_large(self, result: str) -> Tuple[bool, int]:
    """Tokenize the output of the query."""
    if self.tiktoken_model_name:
        tiktoken_ = _import_tiktoken()
        encoding = tiktoken_.encoding_for_model(self.tiktoken_model_name)
        length = len(encoding.encode(result))
        logger.info('Result length: %s', length)
        return (length > self.output_token_limit, length)
    return (False, 0)