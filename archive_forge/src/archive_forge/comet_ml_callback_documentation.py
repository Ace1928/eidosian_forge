import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
Flush the tracker and setup the session.

        Everything after this will be a new table.

        Args:
            name: Name of the performed session so far so it is identifiable
            langchain_asset: The langchain asset to save.
            finish: Whether to finish the run.

            Returns:
                None
        