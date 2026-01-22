from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain.chains.base import Chain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.constitutional_ai.principles import PRINCIPLES
from langchain.chains.constitutional_ai.prompts import CRITIQUE_PROMPT, REVISION_PROMPT
from langchain.chains.llm import LLMChain
@staticmethod
def _parse_critique(output_string: str) -> str:
    if 'Revision request:' not in output_string:
        return output_string
    output_string = output_string.split('Revision request:')[0]
    if '\n\n' in output_string:
        output_string = output_string.split('\n\n')[0]
    return output_string