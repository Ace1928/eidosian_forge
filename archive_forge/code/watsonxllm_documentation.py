import logging
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
Call the IBM watsonx.ai inference endpoint which then streams the response.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The iterator which yields generation chunks.
        Example:
            .. code-block:: python

                response = watsonx_llm.stream("What is a molecule")
                for chunk in response:
                    print(chunk, end='')  # noqa: T201
        