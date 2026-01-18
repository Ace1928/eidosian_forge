from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.outputs import LLMResult
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
Instantiate AsyncFinalIteratorCallbackHandler.

        Args:
            answer_prefix_tokens: Token sequence that prefixes the answer.
                Default is ["Final", "Answer", ":"]
            strip_tokens: Ignore white spaces and new lines when comparing
                answer_prefix_tokens to last tokens? (to determine if answer has been
                reached)
            stream_prefix: Should answer prefix itself also be streamed?
        