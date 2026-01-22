from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain.chains.base import Chain
def _moderate(self, text: str, results: dict) -> str:
    if results['flagged']:
        error_str = "Text was found that violates OpenAI's content policy."
        if self.error:
            raise ValueError(error_str)
        else:
            return error_str
    return text