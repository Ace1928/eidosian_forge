from typing import (
from tenacity import (
from langchain_core.runnables.base import Input, Output, RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
def _patch_config(self, config: RunnableConfig, run_manager: 'T', retry_state: RetryCallState) -> RunnableConfig:
    attempt = retry_state.attempt_number
    tag = 'retry:attempt:{}'.format(attempt) if attempt > 1 else None
    return patch_config(config, callbacks=run_manager.get_child(tag))