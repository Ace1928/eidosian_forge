from __future__ import annotations
import sys
from typing import TYPE_CHECKING, List, Optional, cast
from argparse import ArgumentParser
from typing_extensions import Literal, NamedTuple
from ..._utils import get_client
from ..._models import BaseModel
from ...._streaming import Stream
from ....types.chat import (
from ....types.chat.completion_create_params import (
class CLIMessage(NamedTuple):
    role: ChatCompletionRole
    content: str