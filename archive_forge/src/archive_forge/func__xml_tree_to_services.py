from __future__ import unicode_literals
import sys
import copy
import hashlib
import logging
import os
import tempfile
import warnings
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, REVERSE_TYPE_MAP, Struct
from .transport import get_http_wrapper, set_http_wrapper, get_Http
from .helpers import Alias, fetch, sort_dict, make_key, process_element, \
from .wsse import UsernameToken
def _xml_tree_to_services(self, wsdl, cache, force_download):
    """Convert SimpleXMLElement tree representation of the WSDL into pythonic objects"""
    xsd_ns = None
    soap_uris = {}
    for k, v in wsdl[:]:
        if v in self.soap_ns_uris and k.startswith('xmlns:'):
            soap_uris[get_local_name(k)] = v
        if v == self.xsd_uri and k.startswith('xmlns:'):
            xsd_ns = get_local_name(k)
    elements = {}
    messages = {}
    port_types = {}
    bindings = {}
    services = {}
    if 'http://xml.apache.org/xml-soap' in dict(wsdl[:]).values():
        if wsdl('types', error=False):
            schema = wsdl.types('schema', ns=self.xsd_uri)
            attrs = dict(schema[:])
            self.namespace = attrs.get('targetNamespace', self.namespace)
        if not self.namespace or self.namespace == 'urn:DefaultNamespace':
            self.namespace = wsdl['targetNamespace'] or self.namespace
    imported_schemas = {}
    global_namespaces = {None: self.namespace}
    for types in wsdl('types', error=False) or []:
        schemas = types('schema', ns=self.xsd_uri, error=False)
        for schema in schemas or []:
            preprocess_schema(schema, imported_schemas, elements, self.xsd_uri, self.__soap_server, self.http, cache, force_download, self.wsdl_basedir, global_namespaces=global_namespaces)
    postprocess_element(elements, [])
    for message in wsdl.message:
        for part in message('part', error=False) or []:
            element = {}
            element_name = part['element']
            if not element_name:
                element_name = part['type']
            type_ns = get_namespace_prefix(element_name)
            type_uri = part.get_namespace_uri(type_ns)
            part_name = part['name'] or None
            if type_uri == self.xsd_uri:
                element_name = get_local_name(element_name)
                fn = REVERSE_TYPE_MAP.get(element_name, None)
                element = {part_name: fn}
                if (message['name'], part_name) not in messages:
                    od = Struct()
                    od.namespaces[None] = type_uri
                    messages[message['name'], part_name] = {message['name']: od}
                else:
                    od = messages[message['name'], part_name].values()[0]
                od.namespaces[part_name] = type_uri
                od.references[part_name] = False
                od.update(element)
            else:
                element_name = get_local_name(element_name)
                fn = elements.get(make_key(element_name, 'element', type_uri))
                if not fn:
                    fn = elements.get(make_key(element_name, 'complexType', type_uri))
                    od = Struct()
                    od[part_name] = fn
                    od.namespaces[None] = type_uri
                    od.namespaces[part_name] = type_uri
                    od.references[part_name] = False
                    element = {message['name']: od}
                else:
                    element = {element_name: fn}
                messages[message['name'], part_name] = element
    for port_type_node in wsdl.portType:
        port_type_name = port_type_node['name']
        port_type = port_types[port_type_name] = {}
        operations = port_type['operations'] = {}
        for operation_node in port_type_node.operation:
            op_name = operation_node['name']
            op = operations[op_name] = {}
            op['style'] = operation_node['style']
            op['parameter_order'] = (operation_node['parameterOrder'] or '').split(' ')
            op['documentation'] = unicode(operation_node('documentation', error=False)) or ''
            if operation_node('input', error=False):
                op['input_msg'] = get_local_name(operation_node.input['message'])
                ns = get_namespace_prefix(operation_node.input['message'])
                op['namespace'] = operation_node.get_namespace_uri(ns)
            if operation_node('output', error=False):
                op['output_msg'] = get_local_name(operation_node.output['message'])
            fault_msgs = op['fault_msgs'] = {}
            faults = operation_node('fault', error=False)
            if faults is not None:
                for fault in operation_node('fault', error=False):
                    fault_msgs[fault['name']] = get_local_name(fault['message'])
    for binding_node in wsdl.binding:
        port_type_name = get_local_name(binding_node['type'])
        if port_type_name not in port_types:
            continue
        port_type = port_types[port_type_name]
        binding_name = binding_node['name']
        soap_binding = binding_node('binding', ns=list(soap_uris.values()), error=False)
        transport = soap_binding and soap_binding['transport'] or None
        style = soap_binding and soap_binding['style'] or None
        binding = bindings[binding_name] = {'name': binding_name, 'operations': copy.deepcopy(port_type['operations']), 'port_type_name': port_type_name, 'transport': transport, 'style': style}
        for operation_node in binding_node.operation:
            op_name = operation_node['name']
            op_op = operation_node('operation', ns=list(soap_uris.values()), error=False)
            action = op_op and op_op['soapAction']
            op = binding['operations'].setdefault(op_name, {})
            op['name'] = op_name
            op['style'] = op.get('style', style)
            if action is not None:
                op['action'] = action
            input = operation_node('input', error=False)
            body = input and input('body', ns=list(soap_uris.values()), error=False)
            parts_input_body = body and body['parts'] or None
            parts_input_headers = []
            headers = input and input('header', ns=list(soap_uris.values()), error=False)
            for header in headers or []:
                hdr = {'message': header['message'], 'part': header['part']}
                parts_input_headers.append(hdr)
            if 'input_msg' in op:
                headers = {}
                for input_header in parts_input_headers:
                    header_msg = get_local_name(input_header.get('message'))
                    header_part = get_local_name(input_header.get('part'))
                    hdr = get_message(messages, header_msg or op['input_msg'], header_part)
                    if hdr:
                        headers.update(hdr)
                    else:
                        pass
                op['input'] = get_message(messages, op['input_msg'], parts_input_body, op['parameter_order'])
                op['header'] = headers
                try:
                    element = list(op['input'].values())[0]
                    ns_uri = element.namespaces[None]
                    qualified = element.qualified
                except (AttributeError, KeyError) as e:
                    ns_uri = op['namespace']
                    qualified = None
                if ns_uri:
                    op['namespace'] = ns_uri
                    op['qualified'] = qualified
                del op['input_msg']
            else:
                op['input'] = None
                op['header'] = None
            output = operation_node('output', error=False)
            body = output and output('body', ns=list(soap_uris.values()), error=False)
            parts_output_body = body and body['parts'] or None
            if 'output_msg' in op:
                op['output'] = get_message(messages, op['output_msg'], parts_output_body)
                del op['output_msg']
            else:
                op['output'] = None
            if 'fault_msgs' in op:
                faults = op['faults'] = {}
                for msg in op['fault_msgs'].values():
                    msg_obj = get_message(messages, msg, parts_output_body)
                    tag_name = list(msg_obj)[0]
                    faults[tag_name] = msg_obj
            parts_output_headers = []
            headers = output and output('header', ns=list(soap_uris.values()), error=False)
            for header in headers or []:
                hdr = {'message': header['message'], 'part': header['part']}
                parts_output_headers.append(hdr)
    for service in wsdl('service', error=False) or []:
        service_name = service['name']
        if not service_name:
            continue
        serv = services.setdefault(service_name, {})
        ports = serv['ports'] = {}
        serv['documentation'] = service['documentation'] or ''
        for port in service.port:
            binding_name = get_local_name(port['binding'])
            if not binding_name in bindings:
                continue
            binding = ports[port['name']] = copy.deepcopy(bindings[binding_name])
            address = port('address', ns=list(soap_uris.values()), error=False)
            location = address and address['location'] or None
            soap_uri = address and soap_uris.get(address.get_prefix())
            soap_ver = soap_uri and self.soap_ns_uris.get(soap_uri)
            binding.update({'location': location, 'service_name': service_name, 'soap_uri': soap_uri, 'soap_ver': soap_ver})
    if not services:
        services[''] = {'ports': {'': None}}
    elements = list((e for e in elements.values() if type(e) is type)) + sorted((e for e in elements.values() if not type(e) is type))
    e = None
    self.elements = []
    for element in elements:
        if e != element:
            self.elements.append(element)
        e = element
    return services