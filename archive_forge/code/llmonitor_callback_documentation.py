import importlib.metadata
import logging
import os
import traceback
import warnings
from contextvars import ContextVar
from typing import Any, Dict, List, Union, cast
from uuid import UUID
import requests
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from packaging.version import parse
Callback Handler for LLMonitor`.

    #### Parameters:
        - `app_id`: The app id of the app you want to report to. Defaults to
        `None`, which means that `LLMONITOR_APP_ID` will be used.
        - `api_url`: The url of the LLMonitor API. Defaults to `None`,
        which means that either `LLMONITOR_API_URL` environment variable
        or `https://app.llmonitor.com` will be used.

    #### Raises:
        - `ValueError`: if `app_id` is not provided either as an
        argument or as an environment variable.
        - `ConnectionError`: if the connection to the API fails.


    #### Example:
    ```python
    from langchain_community.llms import OpenAI
    from langchain_community.callbacks import LLMonitorCallbackHandler

    llmonitor_callback = LLMonitorCallbackHandler()
    llm = OpenAI(callbacks=[llmonitor_callback],
                 metadata={"userId": "user-123"})
    llm.invoke("Hello, how are you?")
    ```
    