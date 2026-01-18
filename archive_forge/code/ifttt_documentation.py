from typing import Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
IFTTT Webhook.

    Args:
        name: name of the tool
        description: description of the tool
        url: url to hit with the json event.
    