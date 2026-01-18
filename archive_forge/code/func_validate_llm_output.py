from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.output_parsers.regex import RegexParser
@root_validator()
def validate_llm_output(cls, values: Dict) -> Dict:
    """Validate that the combine chain outputs a dictionary."""
    output_parser = values['llm_chain'].prompt.output_parser
    if not isinstance(output_parser, RegexParser):
        raise ValueError(f'Output parser of llm_chain should be a RegexParser, got {output_parser}')
    output_keys = output_parser.output_keys
    if values['rank_key'] not in output_keys:
        raise ValueError(f'Got {values['rank_key']} as key to rank on, but did not find it in the llm_chain output keys ({output_keys})')
    if values['answer_key'] not in output_keys:
        raise ValueError(f'Got {values['answer_key']} as key to return, but did not find it in the llm_chain output keys ({output_keys})')
    return values