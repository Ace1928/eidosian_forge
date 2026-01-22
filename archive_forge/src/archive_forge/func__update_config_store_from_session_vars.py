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
def _update_config_store_from_session_vars(self, logical_name, config_options):
    config_chain_builder = ConfigChainFactory(session=self._session)
    config_name, env_vars, default, typecast = config_options
    config_store = self._session.get_component('config_store')
    config_store.set_config_provider(logical_name, config_chain_builder.create_config_chain(instance_name=logical_name, env_var_names=env_vars, config_property_names=config_name, default=default, conversion_func=typecast))