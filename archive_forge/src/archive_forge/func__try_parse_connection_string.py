from __future__ import annotations
import string
from queue import Empty
from typing import Any, Dict, Set
import azure.core.exceptions
import azure.servicebus.exceptions
import isodate
from azure.servicebus import (ServiceBusClient, ServiceBusMessage,
from azure.servicebus.management import ServiceBusAdministrationClient
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _try_parse_connection_string(self) -> None:
    self._namespace, self._credential = Transport.parse_uri(self.conninfo.hostname)
    if isinstance(self._credential, DefaultAzureCredential) or isinstance(self._credential, ManagedIdentityCredential):
        return None
    if ':' in self._credential:
        self._policy, self._sas_key = self._credential.split(':', 1)
    conn_dict = {'Endpoint': 'sb://' + self._namespace, 'SharedAccessKeyName': self._policy, 'SharedAccessKey': self._sas_key}
    self._connection_string = ';'.join([key + '=' + value for key, value in conn_dict.items()])