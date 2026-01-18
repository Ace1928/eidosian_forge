from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT, clean_body
Configuration for this pydantic object.