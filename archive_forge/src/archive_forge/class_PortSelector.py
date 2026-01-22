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
class PortSelector:
    """
    The B{port} selector is used to select a I{web service} B{port}.

    In cases where multiple ports have been defined and no default has been
    specified, the port is found by name (or index) and a L{MethodSelector} for
    the port is returned. In all cases, attribute access is forwarded to the
    L{MethodSelector} for either the I{first} port or the I{default} port (when
    specified).

    @ivar __client: A suds client.
    @type __client: L{Client}
    @ivar __ports: A list of I{service} ports.
    @type __ports: list
    @ivar __qn: The I{qualified} name of the port (used for logging).
    @type __qn: str

    """

    def __init__(self, client, ports, qn):
        """
        @param client: A suds client.
        @type client: L{Client}
        @param ports: A list of I{service} ports.
        @type ports: list
        @param qn: The name of the service.
        @type qn: str

        """
        self.__client = client
        self.__ports = ports
        self.__qn = qn

    def __getattr__(self, name):
        """
        Attribute access is forwarded to the L{MethodSelector}.

        Uses the I{default} port when specified or the I{first} port otherwise.

        @param name: The name of a method.
        @type name: str
        @return: A L{MethodSelector}.
        @rtype: L{MethodSelector}.

        """
        default = self.__dp()
        if default is None:
            m = self.__find(0)
        else:
            m = default
        return getattr(m, name)

    def __getitem__(self, name):
        """
        Provides I{port} selection by name (string) or index (integer).

        In cases where only a single port is defined or a I{default} has been
        specified, the request is forwarded to the L{MethodSelector}.

        @param name: The name (or index) of a port.
        @type name: int|str
        @return: A L{MethodSelector} for the specified port.
        @rtype: L{MethodSelector}.

        """
        default = self.__dp()
        if default is None:
            return self.__find(name)
        return default

    def __find(self, name):
        """
        Find a I{port} by name (string) or index (integer).

        @param name: The name (or index) of a port.
        @type name: int|str
        @return: A L{MethodSelector} for the found port.
        @rtype: L{MethodSelector}.

        """
        port = None
        if not self.__ports:
            raise Exception('No ports defined: %s' % (self.__qn,))
        if isinstance(name, int):
            qn = '%s[%d]' % (self.__qn, name)
            try:
                port = self.__ports[name]
            except IndexError:
                raise PortNotFound(qn)
        else:
            qn = '.'.join((self.__qn, name))
            for p in self.__ports:
                if name == p.name:
                    port = p
                    break
        if port is None:
            raise PortNotFound(qn)
        qn = '.'.join((self.__qn, port.name))
        return MethodSelector(self.__client, port.methods, qn)

    def __dp(self):
        """
        Get the I{default} port if defined in the I{options}.

        @return: A L{MethodSelector} for the I{default} port.
        @rtype: L{MethodSelector}.

        """
        dp = self.__client.options.port
        if dp is not None:
            return self.__find(dp)