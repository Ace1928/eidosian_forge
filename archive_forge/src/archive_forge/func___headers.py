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
def __headers(self):
    """
        Get HTTP headers for a HTTP/HTTPS SOAP request.

        @return: A dictionary of header/values.
        @rtype: dict

        """
    action = self.method.soap.action
    if isinstance(action, str):
        action = action.encode('utf-8')
    result = {'Content-Type': 'text/xml; charset=utf-8', 'SOAPAction': action}
    result.update(**self.options.headers)
    log.debug('headers = %s', result)
    return result