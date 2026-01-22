import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import (
from langchain_core.utils import get_from_dict_or_env
class SearxResults(dict):
    """Dict like wrapper around search api results."""
    _data: str = ''

    def __init__(self, data: str):
        """Take a raw result from Searx and make it into a dict like object."""
        json_data = json.loads(data)
        super().__init__(json_data)
        self.__dict__ = self

    def __str__(self) -> str:
        """Text representation of searx result."""
        return self._data

    @property
    def results(self) -> Any:
        """Silence mypy for accessing this field.

        :meta private:
        """
        return self.get('results')

    @property
    def answers(self) -> Any:
        """Helper accessor on the json result."""
        return self.get('answers')