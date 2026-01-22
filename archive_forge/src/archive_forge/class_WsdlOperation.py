import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlOperation(WsdlComponent):
    input = output = None
    soap_operation = None

    def __init__(self, elem, wsdl_document):
        super(WsdlOperation, self).__init__(elem, wsdl_document)
        self.faults = {}
        input_child = elem.find(WSDL_INPUT)
        if input_child is not None:
            self.input = WsdlInput(input_child, wsdl_document)
        output_child = elem.find(WSDL_OUTPUT)
        if output_child is not None:
            self.output = WsdlOutput(output_child, wsdl_document)
        for fault_child in elem.iterfind(WSDL_FAULT):
            fault = WsdlFault(fault_child, wsdl_document)
            if fault.name is None:
                continue
            elif fault.local_name in self.faults:
                msg = 'duplicated fault {!r} for {!r}'
                wsdl_document.parse_error(msg.format(fault.local_name, self))
            self.faults[fault.local_name] = fault
        if input_child is not None and output_child is not None:
            children = self.elem[:]
            input_pos = children.index(input_child)
            output_pos = children.index(output_child)
            if input_pos < output_pos:
                self.transmission = 'request-response'
            else:
                self.transmission = 'solicit-response'
        elif input_child is not None:
            self.transmission = 'one-way'
        elif output_child is not None:
            self.transmission = 'notification'
        else:
            self.transmission = None

    @property
    def key(self):
        return (self.local_name, getattr(self.input, 'local_name', None), getattr(self.output, 'local_name', None))

    @property
    def soap_action(self):
        """The SOAP operation's action URI if any, `None` otherwise."""
        if self.soap_operation is not None:
            return self.soap_operation.get('soapAction')

    @property
    def soap_style(self):
        """The SOAP operation's style if any, `None` otherwise."""
        if self.soap_operation is not None:
            style = self.soap_operation.get('style')
            return style if style in ('rpc', 'document') else 'document'