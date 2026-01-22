import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlBinding(WsdlComponent):
    port_type = None
    'The wsdl:portType definition related to the binding instance.'
    soap_binding = None
    'The SOAP binding element if any, `None` otherwise.'

    def __init__(self, elem, wsdl_document):
        super(WsdlBinding, self).__init__(elem, wsdl_document)
        self.operations = {}
        if wsdl_document.soap_binding:
            self.soap_binding = elem.find(SOAP_BINDING)
            if self.soap_binding is None:
                msg = 'missing soap:binding element for {!r}'
                wsdl_document.parse_error(msg.format(self))
        port_type_name = self._parse_reference(elem, 'type')
        try:
            self.port_type = wsdl_document.maps.port_types[port_type_name]
        except KeyError:
            msg = 'missing port type {!r} for {!r}'
            wsdl_document.parse_error(msg.format(port_type_name, self))
            return
        for op_child in elem.iterfind(WSDL_OPERATION):
            op_name = op_child.get('name')
            if op_name is None:
                continue
            input_child = op_child.find(WSDL_INPUT)
            input_name = None if input_child is None else input_child.get('name')
            output_child = op_child.find(WSDL_OUTPUT)
            output_name = None if output_child is None else output_child.get('name')
            key = (op_name, input_name, output_name)
            if key in self.operations:
                msg = 'duplicated operation {!r} for {!r}'
                wsdl_document.parse_error(msg.format(op_name, self))
            try:
                operation = self.port_type.operations[key]
            except KeyError:
                msg = 'operation {!r} not found for {!r}'
                wsdl_document.parse_error(msg.format(op_name, self))
                continue
            else:
                self.operations[key] = operation
            if wsdl_document.soap_binding:
                operation.soap_operation = op_child.find(SOAP_OPERATION)
            if input_child is not None:
                for body_child in input_child.iterfind(SOAP_BODY):
                    operation.input.soap_body = SoapBody(body_child, wsdl_document)
                    break
                operation.input.soap_headers = [SoapHeader(e, wsdl_document) for e in input_child.iterfind(SOAP_HEADER)]
            if output_child is not None:
                for body_child in output_child.iterfind(SOAP_BODY):
                    operation.output.soap_body = SoapBody(body_child, wsdl_document)
                    break
                operation.output.soap_headers = [SoapHeader(e, wsdl_document) for e in output_child.iterfind(SOAP_HEADER)]
            for fault_child in op_child.iterfind(WSDL_FAULT):
                fault = WsdlFault(fault_child, wsdl_document)
                if fault.name and fault.local_name not in operation.faults:
                    msg = 'missing fault {!r} in {!r}'
                    wsdl_document.parse_error(msg.format(fault.local_name, operation))
                for soap_fault_child in fault_child.iterfind(SOAP_FAULT):
                    fault = SoapFault(soap_fault_child, wsdl_document)
                    if fault.name:
                        try:
                            operation.faults[fault.local_name].soap_fault = fault
                        except KeyError:
                            msg = 'missing fault {!r} in {!r}'
                            wsdl_document.parse_error(msg.format(fault.local_name, operation))

    @property
    def soap_transport(self):
        """The SOAP binding's transport URI if any, `None` otherwise."""
        if self.soap_binding is not None:
            return self.soap_binding.get('transport')

    @property
    def soap_style(self):
        """The SOAP binding's style if any, `None` otherwise."""
        if self.soap_binding is not None:
            style = self.soap_binding.get('style')
            return style if style in ('rpc', 'document') else 'document'