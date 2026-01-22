from typing import Iterator, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
class QuestionAnswer(BaseModel):
    """A question and its answer as a list of facts each one should have a source.
    each sentence contains a body and a list of sources."""
    question: str = Field(..., description='Question that was asked')
    answer: List[FactWithEvidence] = Field(..., description='Body of the answer, each fact should be its separate object with a body and a list of sources')