from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.agent_toolkits.powerbi.prompt import (
from langchain_community.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain_community.utilities.powerbi import PowerBIDataset
Construct a Power BI agent from a Chat LLM and tools.

    If you supply only a toolkit and no Power BI dataset, the same LLM is used for both.
    