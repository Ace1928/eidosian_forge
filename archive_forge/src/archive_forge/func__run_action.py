import json
from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_community.tools.connery.models import Action
from langchain_community.tools.connery.tool import ConneryAction
def _run_action(self, action_id: str, input: Dict[str, str]={}) -> Dict[str, str]:
    """
        Runs the specified Connery Action with the provided input.
        Parameters:
            action_id (str): The ID of the action to run.
            prompt (str): This is a plain English prompt
            with all the information needed to run the action.
            input (Dict[str, str]): The input object expected by the action.
            If provided together with the prompt,
            the input takes precedence over the input specified in the prompt.
        Returns:
            Dict[str, str]: The output of the action.
        """
    response = requests.post(f'{self.runner_url}/v1/actions/{action_id}/run', headers=self._get_headers(), data=json.dumps({'input': input}))
    if not response.ok:
        raise ValueError(f'Failed to run action.Status code: {response.status_code}.Error message: {response.json()['error']['message']}')
    if not response.json()['data']['output']:
        return {}
    else:
        return response.json()['data']['output']