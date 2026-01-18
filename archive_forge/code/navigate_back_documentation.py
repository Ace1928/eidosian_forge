from __future__ import annotations
from typing import Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
Use the tool.