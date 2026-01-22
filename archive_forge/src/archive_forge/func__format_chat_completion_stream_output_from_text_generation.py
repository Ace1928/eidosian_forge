import base64
import io
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
from requests import HTTPError
from huggingface_hub.errors import (
from ..constants import ENDPOINT
from ..utils import (
from ._generated.types import (
def _format_chat_completion_stream_output_from_text_generation(item: TextGenerationStreamOutput, created: int) -> ChatCompletionStreamOutput:
    if item.details is None:
        return ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(role='assistant', content=item.token.text), finish_reason=None, index=0)], created=created)
    else:
        return ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(), finish_reason=item.details.finish_reason, index=0)], created=created)