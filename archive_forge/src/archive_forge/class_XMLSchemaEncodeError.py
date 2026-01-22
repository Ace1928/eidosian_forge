from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaEncodeError(XMLSchemaValidationError):
    """
    Raised when an object is not encodable to an XML data string.

    :param validator: the XSD validator.
    :param obj: the not validated XML data.
    :param encoder: the XML encoder.
    :param reason: the detailed reason of failed validation.
    :param source: the XML resource that contains the error.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    """
    message = 'failed encoding {!r} with {!r}.\n'

    def __init__(self, validator: Union['XsdValidator', Callable[[Any], None]], obj: Any, encoder: Any, reason: Optional[str]=None, source: Optional['XMLResource']=None, namespaces: Optional[NamespacesType]=None) -> None:
        super(XMLSchemaEncodeError, self).__init__(validator, obj, reason, source, namespaces)
        self.encoder = encoder