from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.evaluation.scoring.prompt import (
from langchain.schema import RUN_KEY
class ScoreStringResultOutputParser(BaseOutputParser[dict]):
    """A parser for the output of the ScoreStringEvalChain.

    Attributes:
        _type (str): The type of the output parser.

    """

    @property
    def _type(self) -> str:
        """Return the type of the output parser.

        Returns:
            str: The type of the output parser.

        """
        return 'pairwise_string_result'

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Dict: The parsed output.

        Raises:
            ValueError: If the verdict is invalid.

        """
        match = _FIND_DOUBLE_BRACKETS.search(text)
        if match:
            verdict = match.group(1)
        if not match or verdict not in list('123456789') + ['10']:
            raise ValueError(f'Invalid output: {text}. Output must contain a double bracketed string                 with the verdict between 1 and 10.')
        return {'reasoning': text, 'score': int(verdict)}