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
def _openapi_params_to_json_schema(params: List[Parameter], spec: OpenAPISpec) -> dict:
    properties = {}
    required = []
    for p in params:
        if p.param_schema:
            schema = spec.get_schema(p.param_schema)
        else:
            media_type_schema = list(p.content.values())[0].media_type_schema
            schema = spec.get_schema(media_type_schema)
        if p.description and (not schema.description):
            schema.description = p.description
        properties[p.name] = json.loads(schema.json(exclude_none=True))
        if p.required:
            required.append(p.name)
    return {'type': 'object', 'properties': properties, 'required': required}