import os
import sys
from .json_tools import JSONMixin
from .layer import Layer
from ..io.html import deck_to_html
from ..settings import settings as pydeck_settings
from .view import View
from .view_state import ViewState
from .base_map_provider import BaseMapProvider
from .map_styles import DARK, get_from_map_identifier
def _set_api_keys(self, api_keys: dict=None):
    """Sets API key for base map provider for both HTML embedding and the Jupyter widget"""
    for k in api_keys:
        k and BaseMapProvider(k)
    for provider in BaseMapProvider:
        attr_name = f'{provider.value}_key'
        provider_env_var = f'{provider.name}_API_KEY'
        attr_value = api_keys.get(provider.value) or os.getenv(provider_env_var)
        setattr(self, attr_name, attr_value)
        if has_jupyter_extra():
            setattr(self.deck_widget, attr_name, attr_value)