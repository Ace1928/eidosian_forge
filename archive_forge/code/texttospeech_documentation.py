from __future__ import annotations
import tempfile
from typing import TYPE_CHECKING, Any, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_community.utilities.vertexai import get_client_info
Use the tool.