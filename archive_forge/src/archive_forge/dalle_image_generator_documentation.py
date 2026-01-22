import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import (
from langchain_community.utils.openai import is_openai_v1
Run query through OpenAI and parse result.