import logging
from typing import Any, Optional
from langchain_core.language_models.llms import LLM
from langchain_community.llms.ipex_llm import IpexLLM

        Construct low_bit object from model_id

        Args:

            model_id: Path for the bigdl-llm transformers low-bit model folder.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of BigdlLLM.
        