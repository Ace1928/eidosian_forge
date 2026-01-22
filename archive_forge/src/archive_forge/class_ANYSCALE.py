from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
import openai
from typing_extensions import override
class ANYSCALE(LLM):
    """Accessing ANYSCALE"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        self.client = openai.OpenAI(base_url='https://api.endpoints.anyscale.com/v1', api_key=api_key)

    @override
    def query(self, prompt: str) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': prompt}], max_tokens=MAX_TOKENS)
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'codellama/CodeLlama-34b-Instruct-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'HuggingFaceH4/zephyr-7b-beta']