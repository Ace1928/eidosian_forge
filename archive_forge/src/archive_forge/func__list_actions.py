import json
from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_community.tools.connery.models import Action
from langchain_community.tools.connery.tool import ConneryAction
def _list_actions(self) -> List[Action]:
    """
        Returns the list of actions available in the Connery Runner.
        Returns:
            List[Action]: The list of actions available in the Connery Runner.
        """
    response = requests.get(f'{self.runner_url}/v1/actions', headers=self._get_headers())
    if not response.ok:
        raise ValueError(f'Failed to list actions.Status code: {response.status_code}.Error message: {response.json()['error']['message']}')
    return [Action(**action) for action in response.json()['data']]