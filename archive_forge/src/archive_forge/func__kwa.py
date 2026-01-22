from saml2 import ExtensionElement
from saml2 import SamlBase
from saml2 import element_to_extension_element
from saml2 import extension_elements_to_elements
from saml2 import md
def _kwa(val, onts, mdb_safe=False):
    """
    Key word argument conversion

    :param val: A dictionary
    :param onts: dictionary with schemas to use in the conversion
        schema namespase is the key in the dictionary
    :return: A converted dictionary
    """
    if not mdb_safe:
        return {k: from_dict(v, onts) for k, v in val.items() if k not in EXP_SKIP}
    else:
        _skip = ['_id']
        _skip.extend(EXP_SKIP)
        return {k.replace('__', '.'): from_dict(v, onts) for k, v in val.items() if k not in _skip}