import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
Call the API and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain_community.llms import KoboldApiLLM

                llm = KoboldApiLLM(endpoint="http://localhost:5000")
                llm("Write a story about dragons.")
        