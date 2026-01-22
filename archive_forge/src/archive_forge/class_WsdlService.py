import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlService(WsdlComponent):

    def __init__(self, elem, wsdl_document):
        super(WsdlService, self).__init__(elem, wsdl_document)
        self.ports = {}
        for port_child in elem.iterfind(WSDL_PORT):
            port = WsdlPort(port_child, wsdl_document)
            port_name = port.local_name
            if port_name is None:
                continue
            elif port_name in self.ports:
                msg = 'duplicated port {!r} for {!r}'
                wsdl_document.parse_error(msg.format(port.prefixed_name, self))
            else:
                self.ports[port_name] = port