from typing import Any, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.llms.mlx_pipeline import MLXPipeline
@staticmethod
def _to_chat_result(llm_result: LLMResult) -> ChatResult:
    chat_generations = []
    for g in llm_result.generations[0]:
        chat_generation = ChatGeneration(message=AIMessage(content=g.text), generation_info=g.generation_info)
        chat_generations.append(chat_generation)
    return ChatResult(generations=chat_generations, llm_output=llm_result.llm_output)