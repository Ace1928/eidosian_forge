import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _log_model(self, langchain_asset: Any) -> None:
    model_parameters = self._get_llm_parameters(langchain_asset)
    self.experiment.log_parameters(model_parameters, prefix='model')
    langchain_asset_path = Path(self.temp_dir.name, 'model.json')
    model_name = self.name if self.name else LANGCHAIN_MODEL_NAME
    try:
        if hasattr(langchain_asset, 'save'):
            langchain_asset.save(langchain_asset_path)
            self.experiment.log_model(model_name, str(langchain_asset_path))
    except (ValueError, AttributeError, NotImplementedError) as e:
        if hasattr(langchain_asset, 'save_agent'):
            langchain_asset.save_agent(langchain_asset_path)
            self.experiment.log_model(model_name, str(langchain_asset_path))
        else:
            self.comet_ml.LOGGER.error(f'{e} Could not save Langchain Asset for {langchain_asset.__class__.__name__}')