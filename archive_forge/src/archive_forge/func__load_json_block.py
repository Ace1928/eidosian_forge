import json
import re
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.api.openapi.prompts import REQUEST_TEMPLATE
from langchain.chains.llm import LLMChain
def _load_json_block(self, serialized_block: str) -> str:
    try:
        return json.dumps(json.loads(serialized_block, strict=False))
    except json.JSONDecodeError:
        return 'ERROR serializing request.'