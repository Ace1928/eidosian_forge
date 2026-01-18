import json
import logging
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_pydantic_field_names
Build extra kwargs from additional params that were passed in.