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
def _get_bedrock_services_signal(self, body: dict) -> dict:
    """
        This function checks the response body for an interrupt flag or message that indicates
        whether any of the Bedrock services have intervened in the processing flow. It is
        primarily used to identify modifications or interruptions imposed by these services
        during the request-response cycle with a Large Language Model (LLM).
        """
    if self._guardrails_enabled and self.guardrails.get('trace') and self._is_guardrails_intervention(body):
        return {'signal': True, 'reason': 'GUARDRAIL_INTERVENED', 'trace': body.get(AMAZON_BEDROCK_TRACE_KEY)}
    return {'signal': False, 'reason': None, 'trace': None}