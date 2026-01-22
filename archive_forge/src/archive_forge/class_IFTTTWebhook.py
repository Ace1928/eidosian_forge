from typing import Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
class IFTTTWebhook(BaseTool):
    """IFTTT Webhook.

    Args:
        name: name of the tool
        description: description of the tool
        url: url to hit with the json event.
    """
    url: str

    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        body = {'this': tool_input}
        response = requests.post(self.url, data=body)
        return response.text