import json
import copy
from io import IOBase, TextIOBase
from typing import Any, Dict, List, Optional, Type, Union, Tuple, \
from elementpath.etree import ElementTree, etree_tostring
from .exceptions import XMLSchemaTypeError, XMLSchemaValueError, XMLResourceError
from .names import XSD_NAMESPACE, XSI_TYPE, XSD_SCHEMA
from .aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from .helpers import get_extended_qname, is_etree_document
from .resources import fetch_schema_locations, XMLResource
from .validators import XMLSchema10, XMLSchemaBase, XMLSchemaValidationError
def to_etree(obj: Any, schema: Optional[Union[XMLSchemaBase, SchemaSourceType]]=None, cls: Optional[Type[XMLSchemaBase]]=None, path: Optional[str]=None, validation: str='strict', namespaces: Optional[NamespacesType]=None, use_defaults: bool=True, converter: Optional[ConverterType]=None, unordered: bool=False, **kwargs: Any) -> EncodeType[ElementType]:
    """
    Encodes a data structure/object to an ElementTree's Element.

    :param obj: the Python object that has to be encoded to XML data.
    :param schema: can be a schema instance or a file-like object or a file path or a URL     of a resource or a string containing the schema. If not provided a dummy schema is used.
    :param cls: class to use for building the schema instance (for default uses     :class:`XMLSchema10`).
    :param path: is an optional XPath expression for selecting the element of the schema     that matches the data that has to be encoded. For default the first global element of     the schema is used.
    :param validation: the XSD validation mode. Can be 'strict', 'lax' or 'skip'.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    :param use_defaults: whether to use default values for filling missing data.
    :param converter: an :class:`XMLSchemaConverter` subclass or instance to use for     the encoding.
    :param unordered: a flag for explicitly activating unordered encoding mode for     content model data. This mode uses content models for a reordered-by-model     iteration of the child elements.
    :param kwargs: other optional arguments of :meth:`XMLSchemaBase.iter_encode` and     options for the converter.
    :return: An element tree's Element instance. If ``validation='lax'`` keyword argument is     provided the validation errors are collected and returned coupled in a tuple with the     Element instance.
    :raises: :exc:`XMLSchemaValidationError` if the object is not encodable by the schema,     or also if it's invalid when ``validation='strict'`` is provided.
    """
    if cls is None:
        cls = XMLSchema10
    elif not issubclass(cls, XMLSchemaBase):
        raise XMLSchemaTypeError('invalid schema class %r' % cls)
    if schema is None:
        if not path:
            raise XMLSchemaTypeError('without schema a path is required for building a dummy schema')
        if namespaces is None:
            tag = get_extended_qname(path, {'xsd': XSD_NAMESPACE, 'xs': XSD_NAMESPACE})
        else:
            tag = get_extended_qname(path, namespaces)
        if not tag.startswith('{') and ':' in tag:
            raise XMLSchemaTypeError('without schema the path must be mappable to a local or extended name')
        if tag == XSD_SCHEMA:
            assert cls.meta_schema is not None
            _schema = cls.meta_schema
        else:
            _schema = get_dummy_schema(tag, cls)
    elif isinstance(schema, XMLSchemaBase):
        _schema = schema
    else:
        _schema = cls(schema)
    return _schema.encode(obj=obj, path=path, validation=validation, namespaces=namespaces, use_defaults=use_defaults, converter=converter, unordered=unordered, **kwargs)