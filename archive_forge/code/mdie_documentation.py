from saml2 import ExtensionElement
from saml2 import SamlBase
from saml2 import element_to_extension_element
from saml2 import extension_elements_to_elements
from saml2 import md

    Converts a dictionary into a pysaml2 object
    :param val: A dictionary
    :param onts: Dictionary of schemas to use in the conversion
    :return: The pysaml2 object instance
    