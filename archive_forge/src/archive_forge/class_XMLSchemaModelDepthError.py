from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
class XMLSchemaModelDepthError(XMLSchemaModelError):
    """Raised when recursion depth is exceeded while iterating a model group."""

    def __init__(self, group: 'XsdGroup') -> None:
        msg = 'maximum model recursion depth exceeded while iterating {!r}'.format(group)
        super(XMLSchemaModelDepthError, self).__init__(group, message=msg)