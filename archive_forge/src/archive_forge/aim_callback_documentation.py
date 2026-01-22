from copy import deepcopy
from typing import Any, Dict, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
Flush the tracker and reset the session.

        Args:
            repo (:obj:`str`, optional): Aim repository path or Repo object to which
                Run object is bound. If skipped, default Repo is used.
            experiment_name (:obj:`str`, optional): Sets Run's `experiment` property.
                'default' if not specified. Can be used later to query runs/sequences.
            system_tracking_interval (:obj:`int`, optional): Sets the tracking interval
                in seconds for system usage metrics (CPU, Memory, etc.). Set to `None`
                 to disable system metrics tracking.
            log_system_params (:obj:`bool`, optional): Enable/Disable logging of system
                params such as installed packages, git info, environment variables, etc.
            langchain_asset: The langchain asset to save.
            reset: Whether to reset the session.
            finish: Whether to finish the run.

            Returns:
                None
        