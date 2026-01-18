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
def to_eptid_value(self, values):
    """
        Create AttributeValue instances of NameID from the given values.

        Special handling for the "eptid" attribute
        Name=urn:oid:1.3.6.1.4.1.5923.1.1.1.10
        FriendlyName=eduPersonTargetedID

        values is a list of items of type str or dict. When an item is a
        dictionary it has the keys: "NameQualifier", "SPNameQualifier", and
        "text".

        Returns a list of AttributeValue instances of NameID elements.
        """
    if type(values) is not list:
        values = [values]

    def _create_nameid_ext_el(value):
        text = value['text'] if isinstance(value, dict) else value
        attributes = {'Format': NAMEID_FORMAT_PERSISTENT, 'NameQualifier': value['NameQualifier'], 'SPNameQualifier': value['SPNameQualifier']} if isinstance(value, dict) else {'Format': NAMEID_FORMAT_PERSISTENT}
        element = ExtensionElement('NameID', NAMESPACE, attributes=attributes, text=text)
        return element
    attribute_values = [saml.AttributeValue(extension_elements=[_create_nameid_ext_el(v)]) for v in values]
    return attribute_values