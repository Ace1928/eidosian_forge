from __future__ import annotations
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_core.utils.utils import convert_to_secret_str
class BaseFriendli(Serializable):
    """Base class of Friendli."""
    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)
    model: str = 'mixtral-8x7b-instruct-v0-1'
    friendli_token: Optional[SecretStr] = None
    friendli_team: Optional[str] = None
    streaming: bool = False
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate if personal access token is provided in environment."""
        try:
            import friendli
        except ImportError as e:
            raise ImportError('Could not import friendli-client python package. Please install it with `pip install friendli-client`.') from e
        friendli_token = convert_to_secret_str(get_from_dict_or_env(values, 'friendli_token', 'FRIENDLI_TOKEN'))
        values['friendli_token'] = friendli_token
        friendli_token_str = friendli_token.get_secret_value()
        friendli_team = values['friendli_team'] or os.getenv('FRIENDLI_TEAM')
        values['friendli_team'] = friendli_team
        values['client'] = values['client'] or friendli.Friendli(token=friendli_token_str, team_id=friendli_team)
        values['async_client'] = values['async_client'] or friendli.AsyncFriendli(token=friendli_token_str, team_id=friendli_team)
        return values