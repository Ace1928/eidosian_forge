import inspect
import os
from typing import Any, AnyStr, cast, IO, List, Optional, Type, Union
import maxminddb
from maxminddb import (
import geoip2
import geoip2.models
import geoip2.errors
from geoip2.types import IPAddress
from geoip2.models import (
def _model_for(self, model_class: Union[Type[Country], Type[Enterprise], Type[City]], types: str, ip_address: IPAddress) -> Union[Country, Enterprise, City]:
    record, prefix_len = self._get(types, ip_address)
    traits = record.setdefault('traits', {})
    traits['ip_address'] = ip_address
    traits['prefix_len'] = prefix_len
    return model_class(record, locales=self._locales)