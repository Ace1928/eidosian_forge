from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_community.llms.anthropic import _AnthropicCommon
def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
    """Format a list of messages into a full prompt for the Anthropic model
        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.
        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
    prompt_params = {}
    if self.HUMAN_PROMPT:
        prompt_params['human_prompt'] = self.HUMAN_PROMPT
    if self.AI_PROMPT:
        prompt_params['ai_prompt'] = self.AI_PROMPT
    return convert_messages_to_prompt_anthropic(messages=messages, **prompt_params)