import datetime
from typing import Any, List, Optional
from langchain_core.callbacks import (
from langchain_core.outputs import LLMResult
from langchain_community.llms.openai import OpenAI, OpenAIChat
Call OpenAI generate and then call PromptLayer API to log the request.