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
Call out Friendli's completions API asynchronously with k unique prompts.

        Args:
            prompt (str): The text prompt to generate completion for.
            stop (Optional[List[str]], optional): When one of the stop phrases appears
                in the generation result, the API will stop generation. The stop phrases
                are excluded from the result. If beam search is enabled, all of the
                active beams should contain the stop phrase to terminate generation.
                Before checking whether a stop phrase is included in the result, the
                phrase is converted into tokens. We recommend using stop_tokens because
                it is clearer. For example, after tokenization, phrases "clear" and
                " clear" can result in different token sequences due to the prepended
                space character. Defaults to None.

        Returns:
            str: The generated text output.

        Example:
            .. code-block:: python

                response = await frienldi.agenerate(
                    ["Give me a recipe for the Old Fashioned cocktail."]
                )
        