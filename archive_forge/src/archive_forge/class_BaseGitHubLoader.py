import base64
from abc import ABC
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
class BaseGitHubLoader(BaseLoader, BaseModel, ABC):
    """Load `GitHub` repository Issues."""
    repo: str
    'Name of repository'
    access_token: str
    'Personal access token - see https://github.com/settings/tokens?type=beta'
    github_api_url: str = 'https://api.github.com'
    'URL of GitHub API'

    @root_validator(pre=True, allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that access token exists in environment."""
        values['access_token'] = get_from_dict_or_env(values, 'access_token', 'GITHUB_PERSONAL_ACCESS_TOKEN')
        return values

    @property
    def headers(self) -> Dict[str, str]:
        return {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {self.access_token}'}