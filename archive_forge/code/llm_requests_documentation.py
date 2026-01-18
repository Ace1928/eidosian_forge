from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain.chains import LLMChain
from langchain.chains.base import Chain
Validate that api key and python package exists in environment.