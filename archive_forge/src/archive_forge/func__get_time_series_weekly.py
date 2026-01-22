from typing import Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _get_time_series_weekly(self, symbol: str) -> Dict[str, Any]:
    """Make a request to the AlphaVantage API
        to get the Weekly Time Series."""
    response = requests.get('https://www.alphavantage.co/query/', params={'function': 'TIME_SERIES_WEEKLY', 'symbol': symbol, 'apikey': self.alphavantage_api_key})
    response.raise_for_status()
    data = response.json()
    if 'Error Message' in data:
        raise ValueError(f'API Error: {data['Error Message']}')
    return data