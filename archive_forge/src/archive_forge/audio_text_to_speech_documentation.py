from __future__ import annotations
import logging
from typing import Dict, List, Literal, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field, root_validator, validator
from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool
Use the tool.