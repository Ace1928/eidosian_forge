from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class NutritionAIAPI(BaseModel):
    """Wrapper for the Passio Nutrition AI API."""
    nutritionai_subscription_key: str
    nutritionai_api_url: str = Field(default=DEFAULT_NUTRITIONAI_API_URL)
    more_kwargs: dict = Field(default_factory=dict)
    auth_: ManagedPassioLifeAuth

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @retry(retry=retry_if_result(is_http_retryable), stop=stop_after_attempt(4), wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2))
    def _http_get(self, params: dict) -> requests.Response:
        return requests.get(self.nutritionai_api_url, headers=self.auth_.headers, params=params)

    def _api_call_results(self, search_term: str) -> dict:
        """Call the NutritionAI API and return the results."""
        rsp = self._http_get({'term': search_term, **self.more_kwargs})
        if not rsp:
            raise ValueError('Could not get NutritionAI API results')
        rsp.raise_for_status()
        return rsp.json()

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        nutritionai_subscription_key = get_from_dict_or_env(values, 'nutritionai_subscription_key', 'NUTRITIONAI_SUBSCRIPTION_KEY')
        values['nutritionai_subscription_key'] = nutritionai_subscription_key
        nutritionai_api_url = get_from_dict_or_env(values, 'nutritionai_api_url', 'NUTRITIONAI_API_URL', DEFAULT_NUTRITIONAI_API_URL)
        values['nutritionai_api_url'] = nutritionai_api_url
        values['auth_'] = ManagedPassioLifeAuth(nutritionai_subscription_key)
        return values

    def run(self, query: str) -> Optional[Dict]:
        """Run query through NutrtitionAI API and parse result."""
        results = self._api_call_results(query)
        if results and len(results) < 1:
            return None
        return results