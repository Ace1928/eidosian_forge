from __future__ import annotations
from typing import Dict, List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from ...types import shared_params
from .chat_completion_tool_param import ChatCompletionToolParam
from .chat_completion_message_param import ChatCompletionMessageParam
from .chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from .chat_completion_function_call_option_param import ChatCompletionFunctionCallOptionParam
class CompletionCreateParamsNonStreaming(CompletionCreateParamsBase):
    stream: Optional[Literal[False]]
    'If set, partial message deltas will be sent, like in ChatGPT.\n\n    Tokens will be sent as data-only\n    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)\n    as they become available, with the stream terminated by a `data: [DONE]`\n    message.\n    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).\n    '