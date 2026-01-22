import suds
from suds import *
import suds.bindings.binding
from suds.builder import Builder
import suds.cache
import suds.metrics as metrics
from suds.options import Options
from suds.plugin import PluginContainer
from suds.properties import Unskin
from suds.reader import DefinitionsReader
from suds.resolver import PathResolver
from suds.sax.document import Document
import suds.sax.parser
from suds.servicedefinition import ServiceDefinition
import suds.transport
import suds.transport.https
from suds.umx.basic import Basic as UmxBasic
from suds.wsdl import Definitions
from . import sudsobject
from http.cookiejar import CookieJar
from copy import deepcopy
import http.client
from logging import getLogger
def __get_fault(self, replyroot):
    """
        Extract fault information from a SOAP reply.

        Returns an I{unmarshalled} fault L{Object} or None in case the given
        XML document does not contain a SOAP <Fault> element.

        @param replyroot: A SOAP reply message root XML element or None.
        @type replyroot: L{Element}|I{None}
        @return: A fault object.
        @rtype: L{Object}

        """
    envns = suds.bindings.binding.envns
    soapenv = replyroot and replyroot.getChild('Envelope', envns)
    soapbody = soapenv and soapenv.getChild('Body', envns)
    fault = soapbody and soapbody.getChild('Fault', envns)
    return fault is not None and UmxBasic().process(fault)