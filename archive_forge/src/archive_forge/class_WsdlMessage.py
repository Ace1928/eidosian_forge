import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlMessage(WsdlComponent):

    def __init__(self, elem, wsdl_document):
        super(WsdlMessage, self).__init__(elem, wsdl_document)
        self.parts = {}
        xsd_elements = wsdl_document.schema.maps.elements
        xsd_types = wsdl_document.schema.maps.types
        for child in elem.iterfind(WSDL_PART):
            part_name = child.get('name')
            if part_name is None:
                continue
            elif part_name in self.parts:
                msg = 'duplicated part {!r} for {!r}'
                wsdl_document.parse_error(msg.format(part_name, self))
            try:
                element_attr = child.attrib['element']
            except KeyError:
                pass
            else:
                if 'type' in child.attrib:
                    msg = "ambiguous binding with both 'type' and 'element' attributes"
                    wsdl_document.parse_error(msg)
                element_name = get_extended_qname(element_attr, wsdl_document.namespaces)
                try:
                    self.parts[part_name] = xsd_elements[element_name]
                except KeyError:
                    self.parts[part_name] = xsd_types[XSD_ANY_TYPE]
                    msg = 'missing schema element {!r}'.format(element_name)
                    wsdl_document.parse_error(msg)
                continue
            try:
                type_attr = child.attrib['type']
            except KeyError:
                msg = "missing both 'type' and 'element' attributes"
                wsdl_document.parse_error(msg)
            else:
                type_name = get_extended_qname(type_attr, wsdl_document.namespaces)
                try:
                    self.parts[part_name] = xsd_types[type_name]
                except KeyError:
                    self.parts[part_name] = xsd_types[XSD_ANY_TYPE]
                    msg = 'missing schema type {!r}'.format(type_name)
                    wsdl_document.parse_error(msg)