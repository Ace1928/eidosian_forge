import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _create_session_analysis_dataframe(self, langchain_asset: Any=None) -> dict:
    pd = import_pandas()
    llm_parameters = self._get_llm_parameters(langchain_asset)
    num_generations_per_prompt = llm_parameters.get('n', 1)
    llm_start_records_df = pd.DataFrame(self.on_llm_start_records)
    llm_start_records_df = llm_start_records_df.loc[llm_start_records_df.index.repeat(num_generations_per_prompt)].reset_index(drop=True)
    llm_end_records_df = pd.DataFrame(self.on_llm_end_records)
    llm_session_df = pd.merge(llm_start_records_df, llm_end_records_df, left_index=True, right_index=True, suffixes=['_llm_start', '_llm_end'])
    return llm_session_df