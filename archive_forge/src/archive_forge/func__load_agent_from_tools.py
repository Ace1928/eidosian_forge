import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import yaml
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.types import AGENT_TO_CLASS
from langchain.chains.loading import load_chain, load_chain_from_config
def _load_agent_from_tools(config: dict, llm: BaseLanguageModel, tools: List[Tool], **kwargs: Any) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    config_type = config.pop('_type')
    if config_type not in AGENT_TO_CLASS:
        raise ValueError(f'Loading {config_type} agent not supported')
    agent_cls = AGENT_TO_CLASS[config_type]
    combined_config = {**config, **kwargs}
    return agent_cls.from_llm_and_tools(llm, tools, **combined_config)