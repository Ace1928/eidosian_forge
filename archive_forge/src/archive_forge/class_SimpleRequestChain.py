from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities.openapi import OpenAPISpec
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.input import get_colored_text
from requests import Response
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.tools import APIOperation
class SimpleRequestChain(Chain):
    """Chain for making a simple request to an API endpoint."""
    request_method: Callable
    'Method to use for making the request.'
    output_key: str = 'response'
    'Key to use for the output of the request.'
    input_key: str = 'function'
    'Key to use for the input of the request.'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Run the logic of this chain and return the output."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        name = inputs[self.input_key].pop('name')
        args = inputs[self.input_key].pop('arguments')
        _pretty_name = get_colored_text(name, 'green')
        _pretty_args = get_colored_text(json.dumps(args, indent=2), 'green')
        _text = f'Calling endpoint {_pretty_name} with arguments:\n' + _pretty_args
        _run_manager.on_text(_text)
        api_response: Response = self.request_method(name, args)
        if api_response.status_code != 200:
            response = f'{api_response.status_code}: {api_response.reason}' + f'\nFor {name} ' + f'Called with args: {args.get('params', '')}'
        else:
            try:
                response = api_response.json()
            except Exception:
                response = api_response.text
        return {self.output_key: response}