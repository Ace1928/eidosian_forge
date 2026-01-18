import importlib.util
import logging
from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.self_hosted import SelfHostedPipeline
from langchain_community.llms.utils import enforce_stop_tokens
Get the identifying parameters.