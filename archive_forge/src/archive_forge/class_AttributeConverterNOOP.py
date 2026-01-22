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
class AttributeConverterNOOP(AttributeConverter):
    """Does a NOOP conversion, that is no conversion is made"""

    def __init__(self, name_format=''):
        AttributeConverter.__init__(self, name_format)

    def to_(self, attrvals):
        """Create a list of Attribute instances.

        :param attrvals: A dictionary of attributes and values
        :return: A list of Attribute instances
        """
        attributes = []
        for key, value in attrvals.items():
            key = key.lower()
            attributes.append(factory(saml.Attribute, name=key, name_format=self.name_format, attribute_value=do_ava(value)))
        return attributes