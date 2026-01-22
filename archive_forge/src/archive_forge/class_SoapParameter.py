import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class SoapParameter(WsdlComponent):

    @property
    def use(self):
        use = self.elem.get('use')
        return use if use in ('literal', 'encoded') else None

    @property
    def encoding_style(self):
        return self.elem.get('encodingStyle')

    @property
    def namespace(self):
        return self.elem.get('namespace', '')