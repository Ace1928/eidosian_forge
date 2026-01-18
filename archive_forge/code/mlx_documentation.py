from typing import Any, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.llms.mlx_pipeline import MLXPipeline
Convert LangChain message to ChatML format.