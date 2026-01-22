from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Optional, cast
from argparse import ArgumentParser
from functools import partial
from openai.types.completion import Completion
from .._utils import get_client
from ..._types import NOT_GIVEN, NotGivenOr
from ..._utils import is_given
from .._errors import CLIError
from .._models import BaseModel
from ..._streaming import Stream
class CLICompletionCreateArgs(BaseModel):
    model: str
    stream: bool = False
    prompt: Optional[str] = None
    n: NotGivenOr[int] = NOT_GIVEN
    stop: NotGivenOr[str] = NOT_GIVEN
    user: NotGivenOr[str] = NOT_GIVEN
    echo: NotGivenOr[bool] = NOT_GIVEN
    suffix: NotGivenOr[str] = NOT_GIVEN
    best_of: NotGivenOr[int] = NOT_GIVEN
    top_p: NotGivenOr[float] = NOT_GIVEN
    logprobs: NotGivenOr[int] = NOT_GIVEN
    max_tokens: NotGivenOr[int] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    presence_penalty: NotGivenOr[float] = NOT_GIVEN
    frequency_penalty: NotGivenOr[float] = NOT_GIVEN