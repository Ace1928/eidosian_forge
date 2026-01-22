from typing import Iterator, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
class FactWithEvidence(BaseModel):
    """Class representing a single statement.

    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart
    such that each one only uses a set of sources that are relevant to it.
    """
    fact: str = Field(..., description='Body of the sentence, as part of a response')
    substring_quote: List[str] = Field(..., description='Each source should be a direct quote from the context, as a substring of the original content')

    def _get_span(self, quote: str, context: str, errs: int=100) -> Iterator[str]:
        import regex
        minor = quote
        major = context
        errs_ = 0
        s = regex.search(f'({minor}){{e<={errs_}}}', major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f'({minor}){{e<={errs_}}}', major)
        if s is not None:
            yield from s.spans()

    def get_spans(self, context: str) -> Iterator[str]:
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)