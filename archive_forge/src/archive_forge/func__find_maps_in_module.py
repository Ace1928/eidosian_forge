from importlib import import_module
import logging
import os
import sys
from saml2 import NAMESPACE
from saml2 import ExtensionElement
from saml2 import SAMLError
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2.s_utils import do_ava
from saml2.s_utils import factory
from saml2.saml import NAME_FORMAT_UNSPECIFIED
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def _find_maps_in_module(module):
    """Find attribute map dictionaries in a map file

    :param: module: the python map module
    :type: types.ModuleType
    :return: a generator yielding dict objects which have the right shape
    :rtype: typing.Iterable[dict]
    """
    for key, item in module.__dict__.items():
        if key.startswith('__'):
            continue
        if isinstance(item, dict) and 'identifier' in item and ('to' in item or 'fro' in item):
            yield item