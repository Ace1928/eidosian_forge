import os
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
from langchain_core.outputs import LLMResult

    Callback handler for Trubrics.

    Args:
        project: a trubrics project, default project is "default"
        email: a trubrics account email, can equally be set in env variables
        password: a trubrics account password, can equally be set in env variables
        **kwargs: all other kwargs are parsed and set to trubrics prompt variables,
            or added to the `metadata` dict
    