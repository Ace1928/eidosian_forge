import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlMessageReference(WsdlComponent):
    message = None

    def __init__(self, elem, wsdl_document):
        super(WsdlMessageReference, self).__init__(elem, wsdl_document)
        message_name = self._parse_reference(elem, 'message')
        try:
            self.message = wsdl_document.maps.messages[message_name]
        except KeyError:
            if message_name:
                msg = 'unknown message {!r} for {!r}'
                wsdl_document.parse_error(msg.format(message_name, self))