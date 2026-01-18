from enum import Enum
from typing import Any, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.llms.utils import enforce_stop_tokens
Call out to Titan Takeoff (Pro) stream endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use when streaming.

        Yields:
            A dictionary like object containing a string token.

        Example:
            .. code-block:: python

                model = TitanTakeoff()

                prompt = "What is the capital of the United Kingdom?"
                response = model.stream(prompt)

                # OR

                model = TitanTakeoff(streaming=True)

                response = model.invoke(prompt)

        