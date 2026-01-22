import asyncio
import json
import warnings
from abc import ABC
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.anthropic import (
def _get_guardrails_canonical(self) -> Dict[str, Any]:
    """
        The canonical way to pass in guardrails to the bedrock service
        adheres to the following format:

        "amazon-bedrock-guardrailDetails": {
            "guardrailId": "string",
            "guardrailVersion": "string"
        }
        """
    return {'amazon-bedrock-guardrailDetails': {'guardrailId': self.guardrails.get('id'), 'guardrailVersion': self.guardrails.get('version')}}