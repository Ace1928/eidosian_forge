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
def ac_factory(path=''):
    """Attribute Converter factory

    :param path: The path to a directory where the attribute maps are expected
        to reside.
    :return: A list of AttributeConverter instances
    """
    acs = []
    if path:
        if path not in sys.path:
            sys.path.insert(0, path)
        for fil in sorted(os.listdir(path)):
            if fil.endswith('.py'):
                mod = import_module(fil[:-3])
                acs.extend(_attribute_map_module_to_acs(mod))
    else:
        from saml2 import attributemaps
        for typ in attributemaps.__all__:
            mod = import_module(f'.{typ}', 'saml2.attributemaps')
            acs.extend(_attribute_map_module_to_acs(mod))
    return acs