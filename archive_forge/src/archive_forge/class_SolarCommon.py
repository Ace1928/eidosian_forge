from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
class SolarCommon(BaseModel):
    """Common configuration for Solar LLMs."""
    _client: _SolarClient
    base_url: str = SOLAR_SERVICE_URL_BASE
    solar_api_key: Optional[SecretStr] = Field(default=None, alias='api_key')
    'Solar API key. Get it here: https://console.upstage.ai/services/solar'
    model_name: str = Field(default='solar-1-mini-chat', alias='model')
    'Model name. Available models listed here: https://console.upstage.ai/services/solar'
    max_tokens: int = Field(default=1024)
    temperature = 0.3

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        extra = 'ignore'

    @property
    def lc_secrets(self) -> dict:
        return {'solar_api_key': 'SOLAR_API_KEY'}

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {'model': self.model_name, 'max_tokens': self.max_tokens, 'temperature': self.temperature}

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{'model': self.model_name}, **self._default_params}

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(values, 'solar_api_key', 'SOLAR_API_KEY')
        if api_key is None or len(api_key) == 0:
            raise ValueError('SOLAR_API_KEY must be configured')
        values['solar_api_key'] = convert_to_secret_str(api_key)
        if 'base_url' not in values:
            values['base_url'] = SOLAR_SERVICE_URL_BASE
        if 'base_url' in values and (not values['base_url'].startswith(SOLAR_SERVICE)):
            raise ValueError('base_url must match with: ' + SOLAR_SERVICE)
        values['_client'] = _SolarClient(api_key=values['solar_api_key'], base_url=values['base_url'])
        return values

    @property
    def _llm_type(self) -> str:
        return 'solar'