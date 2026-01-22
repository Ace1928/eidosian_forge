import json
import re
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.api.openapi.prompts import RESPONSE_TEMPLATE
from langchain.chains.llm import LLMChain
class APIResponderChain(LLMChain):
    """Get the response parser."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool=True, **kwargs: Any) -> LLMChain:
        """Get the response parser."""
        output_parser = APIResponderOutputParser()
        prompt = PromptTemplate(template=RESPONSE_TEMPLATE, output_parser=output_parser, input_variables=['response', 'instructions'])
        return cls(prompt=prompt, llm=llm, verbose=verbose, **kwargs)