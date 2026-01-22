import ipaddress
import json
from typing import Any, Dict, cast, List, Optional, Type, Union
import aiohttp
import aiohttp.http
import requests
import requests.utils
import geoip2
import geoip2.models
from geoip2.errors import (
from geoip2.models import City, Country, Insights
from geoip2.types import IPAddress
def _response_for(self, path: str, model_class: Union[Type[Insights], Type[City], Type[Country]], ip_address: IPAddress) -> Union[Country, City, Insights]:
    uri = self._uri(path, ip_address)
    response = self._session.get(uri, proxies=self._proxies, timeout=self._timeout)
    status = response.status_code
    content_type = response.headers['Content-Type']
    body = response.text
    if status != 200:
        raise self._exception_for_error(status, content_type, body, uri)
    decoded_body = self._handle_success(body, uri)
    return model_class(decoded_body, locales=self._locales)