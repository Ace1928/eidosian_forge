import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class SoapBody(SoapParameter):
    """Class for soap:body bindings."""

    def __init__(self, elem, wsdl_document):
        super(SoapBody, self).__init__(elem, wsdl_document)
        self.parts = elem.get('parts', '').split()