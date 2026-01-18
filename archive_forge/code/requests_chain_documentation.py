import json
import re
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.api.openapi.prompts import REQUEST_TEMPLATE
from langchain.chains.llm import LLMChain
Get the request parser.