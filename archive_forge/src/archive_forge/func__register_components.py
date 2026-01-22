import copy
import logging
import os
import platform
import socket
import warnings
import botocore.client
import botocore.configloader
import botocore.credentials
import botocore.tokens
from botocore import (
from botocore.compat import HAS_CRT, MutableMapping
from botocore.configprovider import (
from botocore.errorfactory import ClientExceptionsFactory
from botocore.exceptions import (
from botocore.hooks import (
from botocore.loaders import create_loader
from botocore.model import ServiceModel
from botocore.parsers import ResponseParserFactory
from botocore.regions import EndpointResolver
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.compat import HAS_CRT  # noqa
def _register_components(self):
    self._register_credential_provider()
    self._register_token_provider()
    self._register_data_loader()
    self._register_endpoint_resolver()
    self._register_event_emitter()
    self._register_response_parser_factory()
    self._register_exceptions_factory()
    self._register_config_store()
    self._register_monitor()
    self._register_default_config_resolver()
    self._register_smart_defaults_factory()
    self._register_user_agent_creator()