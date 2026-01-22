from __future__ import annotations
from typing import Any, Dict, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
def answer_cs(self, cs_token: str, OAI_token: str, query: str, apiKey: str) -> dict:
    """
        Send a query to the Cogniswitch service and retrieve the response.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            apiKey (str): OAuth token.
            query (str): Query to be answered.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
    if not cs_token:
        raise ValueError('Missing cs_token')
    if not OAI_token:
        raise ValueError('Missing OpenAI token')
    if not apiKey:
        raise ValueError('Missing cogniswitch OAuth token')
    if not query:
        raise ValueError('Missing input query')
    headers = {'apiKey': apiKey, 'platformToken': cs_token, 'openAIToken': OAI_token}
    data = {'query': query}
    response = requests.post(self.api_url, headers=headers, verify=False, data=data)
    return response.json()