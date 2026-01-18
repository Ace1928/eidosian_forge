from __future__ import unicode_literals
import sys
import datetime
import sys
import logging
import warnings
import re
import traceback
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement, TYPE_MAP, Date, Decimal
def wsdl(self):
    """Generate Web Service Description v1.1"""
    xml = '<?xml version="1.0"?>\n<wsdl:definitions name="%(name)s"\n          targetNamespace="%(namespace)s"\n          xmlns:tns="%(namespace)s"\n          xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"\n          xmlns:wsdl="http://schemas.xmlsoap.org/wsdl/"\n          xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n    <wsdl:documentation xmlns:wsdl="http://schemas.xmlsoap.org/wsdl/">%(documentation)s</wsdl:documentation>\n\n    <wsdl:types>\n       <xsd:schema targetNamespace="%(namespace)s"\n              elementFormDefault="qualified"\n              xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n       </xsd:schema>\n    </wsdl:types>\n\n</wsdl:definitions>\n' % {'namespace': self.namespace, 'name': self.name, 'documentation': self.documentation}
    wsdl = SimpleXMLElement(xml)
    for method, (function, returns, args, doc) in self.methods.items():

        def parse_element(name, values, array=False, complex=False):
            if not complex:
                element = wsdl('wsdl:types')('xsd:schema').add_child('xsd:element')
                complex = element.add_child('xsd:complexType')
            else:
                complex = wsdl('wsdl:types')('xsd:schema').add_child('xsd:complexType')
                element = complex
            element['name'] = name
            if values:
                items = values
            elif values is None:
                items = [('value', None)]
            else:
                items = []
            if not array and items:
                all = complex.add_child('xsd:all')
            elif items:
                all = complex.add_child('xsd:sequence')
            for k, v in items:
                e = all.add_child('xsd:element')
                e['name'] = k
                if array:
                    e[:] = {'minOccurs': '0', 'maxOccurs': 'unbounded'}
                if v in TYPE_MAP.keys():
                    t = 'xsd:%s' % TYPE_MAP[v]
                elif v is None:
                    t = 'xsd:anyType'
                elif isinstance(v, list):
                    n = 'ArrayOf%s%s' % (name, k)
                    l = []
                    for d in v:
                        l.extend(d.items())
                    parse_element(n, l, array=True, complex=True)
                    t = 'tns:%s' % n
                elif isinstance(v, dict):
                    n = '%s%s' % (name, k)
                    parse_element(n, v.items(), complex=True)
                    t = 'tns:%s' % n
                else:
                    raise TypeError('unknonw type %s for marshalling' % str(v))
                e.add_attribute('type', t)
        parse_element('%s' % method, args and args.items())
        parse_element('%sResponse' % method, returns and returns.items())
        for m, e in (('Input', ''), ('Output', 'Response')):
            message = wsdl.add_child('wsdl:message')
            message['name'] = '%s%s' % (method, m)
            part = message.add_child('wsdl:part')
            part[:] = {'name': 'parameters', 'element': 'tns:%s%s' % (method, e)}
    portType = wsdl.add_child('wsdl:portType')
    portType['name'] = '%sPortType' % self.name
    for method, (function, returns, args, doc) in self.methods.items():
        op = portType.add_child('wsdl:operation')
        op['name'] = method
        if doc:
            op.add_child('wsdl:documentation', doc)
        input = op.add_child('wsdl:input')
        input['message'] = 'tns:%sInput' % method
        output = op.add_child('wsdl:output')
        output['message'] = 'tns:%sOutput' % method
    binding = wsdl.add_child('wsdl:binding')
    binding['name'] = '%sBinding' % self.name
    binding['type'] = 'tns:%sPortType' % self.name
    soapbinding = binding.add_child('soap:binding')
    soapbinding['style'] = 'document'
    soapbinding['transport'] = 'http://schemas.xmlsoap.org/soap/http'
    for method in self.methods.keys():
        op = binding.add_child('wsdl:operation')
        op['name'] = method
        soapop = op.add_child('soap:operation')
        soapop['soapAction'] = self.action + method
        soapop['style'] = 'document'
        input = op.add_child('wsdl:input')
        soapbody = input.add_child('soap:body')
        soapbody['use'] = 'literal'
        output = op.add_child('wsdl:output')
        soapbody = output.add_child('soap:body')
        soapbody['use'] = 'literal'
    service = wsdl.add_child('wsdl:service')
    service['name'] = '%sService' % self.name
    service.add_child('wsdl:documentation', text=self.documentation)
    port = service.add_child('wsdl:port')
    port['name'] = '%s' % self.name
    port['binding'] = 'tns:%sBinding' % self.name
    soapaddress = port.add_child('soap:address')
    soapaddress['location'] = self.location
    return wsdl.as_xml(pretty=True)