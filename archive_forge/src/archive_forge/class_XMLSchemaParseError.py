from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaParseError(XMLSchemaValidatorError, SyntaxError):
    """
    Raised when an error is found during the building of an XSD validator.

    :param validator: the XSD validator.
    :param message: the error message.
    :param elem: the element that contains the error.
    """

    def __init__(self, validator: 'XsdValidator', message: str, elem: Optional[ElementType]=None) -> None:
        super(XMLSchemaParseError, self).__init__(validator=validator, message=message, elem=elem if elem is not None else getattr(validator, 'elem', None), source=getattr(validator, 'source', None), namespaces=getattr(validator, 'namespaces', None))