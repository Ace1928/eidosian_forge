from __future__ import annotations
from typing import List, Optional, Sequence
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.utils.input import get_color_mapping, print_text
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
Compare model outputs on an input text.

        If a prompt was provided with starting the laboratory, then this text will be
        fed into the prompt. If no prompt was provided, then the input text is the
        entire prompt.

        Args:
            text: input text to run all models on.
        