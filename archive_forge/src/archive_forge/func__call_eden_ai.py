from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _call_eden_ai(self, query_params: Dict[str, Any]) -> str:
    """
        Make an API call to the EdenAI service with the specified query parameters.

        Args:
            query_params (dict): The parameters to include in the API call.

        Returns:
            requests.Response: The response from the EdenAI API call.

        """
    headers = {'Authorization': f'Bearer {self.edenai_api_key}', 'User-Agent': self.get_user_agent()}
    url = f'https://api.edenai.run/v2/{self.feature}/{self.subfeature}'
    payload = {'providers': str(self.providers), 'response_as_dict': False, 'attributes_as_list': True, 'show_original_response': False}
    payload.update(query_params)
    response = requests.post(url, json=payload, headers=headers)
    self._raise_on_error(response)
    try:
        return self._parse_response(response.json())
    except Exception as e:
        raise RuntimeError(f'An error occurred while running tool: {e}')