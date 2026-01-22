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
class SubsetChainConfigFactory:
    """A class for creating backwards compatible configuration chains.

    This class can be used instead of
    :class:`botocore.configprovider.ConfigChainFactory` to make it honor the
    methods argument to get_config_variable. This class can be used to filter
    out providers that are not in the methods tuple when creating a new config
    chain.
    """

    def __init__(self, session, methods, environ=None):
        self._factory = ConfigChainFactory(session, environ)
        self._supported_methods = methods

    def create_config_chain(self, instance_name=None, env_var_names=None, config_property_name=None, default=None, conversion_func=None):
        """Build a config chain following the standard botocore pattern.

        This config chain factory will omit any providers not in the methods
        tuple provided at initialization. For example if given the tuple
        ('instance', 'config',) it will not inject the environment provider
        into the standard config chain. This lets the botocore session support
        the custom ``methods`` argument for all the default botocore config
        variables when calling ``get_config_variable``.
        """
        if 'instance' not in self._supported_methods:
            instance_name = None
        if 'env' not in self._supported_methods:
            env_var_names = None
        if 'config' not in self._supported_methods:
            config_property_name = None
        return self._factory.create_config_chain(instance_name=instance_name, env_var_names=env_var_names, config_property_names=config_property_name, default=default, conversion_func=conversion_func)