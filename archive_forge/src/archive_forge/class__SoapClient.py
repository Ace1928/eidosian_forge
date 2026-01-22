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
class _SoapClient:
    """
    An internal lightweight SOAP based web service operation client.

    Each instance is constructed for specific web service operation and knows
    how to:
      - Construct a SOAP request for it.
      - Transport a SOAP request for it using a configured transport.
      - Receive a SOAP reply using a configured transport.
      - Process the received SOAP reply.

    Depending on the given suds options, may do all the tasks listed above or
    may stop the process at an earlier point and return some intermediate
    result, e.g. the constructed SOAP request or the raw received SOAP reply.
    See the invoke() method for more detailed information.

    @ivar service: The target method.
    @type service: L{Service}
    @ivar method: A target method.
    @type method: L{Method}
    @ivar options: A dictonary of options.
    @type options: dict
    @ivar cookiejar: A cookie jar.
    @type cookiejar: libcookie.CookieJar

    """
    TIMEOUT_ARGUMENT = '__timeout'

    def __init__(self, client, method):
        """
        @param client: A suds client.
        @type client: L{Client}
        @param method: A target method.
        @type method: L{Method}

        """
        self.client = client
        self.method = method
        self.options = client.options
        self.cookiejar = CookieJar()

    def invoke(self, args, kwargs):
        """
        Invoke a specified web service method.

        Depending on how the ``nosend`` & ``retxml`` options are set, may do
        one of the following:
          * Return a constructed web service operation SOAP request without
            sending it to the web service.
          * Invoke the web service operation and return its SOAP reply XML.
          * Invoke the web service operation, process its results and return
            the Python object representing the returned value.

        When returning a SOAP request, the request is wrapped inside a
        RequestContext object allowing the user to acquire a corresponding SOAP
        reply himself and then pass it back to suds for further processing.

        Constructed request data is automatically processed using registered
        plugins and serialized into a byte-string. Exact request XML formatting
        may be affected by the ``prettyxml`` suds option.

        @param args: A list of args for the method invoked.
        @type args: list|tuple
        @param kwargs: Named (keyword) args for the method invoked.
        @type kwargs: dict
        @return: SOAP request, SOAP reply or a web service return value.
        @rtype: L{RequestContext}|I{builtin}|I{subclass of} L{Object}|I{bytes}|
            I{None}

        """
        timer = metrics.Timer()
        timer.start()
        binding = self.method.binding.input
        timeout = kwargs.pop(_SoapClient.TIMEOUT_ARGUMENT, None)
        soapenv = binding.get_message(self.method, args, kwargs)
        timer.stop()
        method_name = self.method.name
        metrics.log.debug("message for '%s' created: %s", method_name, timer)
        timer.start()
        result = self.send(soapenv, timeout=timeout)
        timer.stop()
        metrics.log.debug("method '%s' invoked: %s", method_name, timer)
        return result

    def send(self, soapenv, timeout=None):
        """
        Send SOAP message.

        Depending on how the ``nosend`` & ``retxml`` options are set, may do
        one of the following:
          * Return a constructed web service operation request without sending
            it to the web service.
          * Invoke the web service operation and return its SOAP reply XML.
          * Invoke the web service operation, process its results and return
            the Python object representing the returned value.

        @param soapenv: A SOAP envelope to send.
        @type soapenv: L{Document}
        @return: SOAP request, SOAP reply or a web service return value.
        @rtype: L{RequestContext}|I{builtin}|I{subclass of} L{Object}|I{bytes}|
            I{None}

        """
        location = self.__location()
        log.debug('sending to (%s)\nmessage:\n%s', location, soapenv)
        self.last_sent(soapenv)
        plugins = PluginContainer(self.options.plugins)
        plugins.message.marshalled(envelope=soapenv.root())
        if self.options.prettyxml:
            soapenv = soapenv.str()
        else:
            soapenv = soapenv.plain()
        soapenv = soapenv.encode('utf-8')
        ctx = plugins.message.sending(envelope=soapenv)
        soapenv = ctx.envelope
        if self.options.nosend:
            return RequestContext(self.process_reply, soapenv)
        request = suds.transport.Request(location, soapenv, timeout)
        request.headers = self.__headers()
        try:
            timer = metrics.Timer()
            timer.start()
            reply = self.options.transport.send(request)
            timer.stop()
            metrics.log.debug('waited %s on server reply', timer)
        except suds.transport.TransportError as e:
            content = e.fp and e.fp.read() or ''
            return self.process_reply(content, e.httpcode, tostr(e))
        return self.process_reply(reply.message, None, None)

    def process_reply(self, reply, status, description):
        """
        Process a web service operation SOAP reply.

        Depending on how the ``retxml`` option is set, may return the SOAP
        reply XML or process it and return the Python object representing the
        returned value.

        @param reply: The SOAP reply envelope.
        @type reply: I{bytes}
        @param status: The HTTP status code (None indicates httplib.OK).
        @type status: int|I{None}
        @param description: Additional status description.
        @type description: str
        @return: The invoked web service operation return value.
        @rtype: I{builtin}|I{subclass of} L{Object}|I{bytes}|I{None}

        """
        if status is None:
            status = http.client.OK
        debug_message = 'Reply HTTP status - %d' % (status,)
        if status in (http.client.ACCEPTED, http.client.NO_CONTENT):
            log.debug(debug_message)
            return
        if status == http.client.OK:
            log.debug('%s\n%s', debug_message, reply)
        else:
            log.debug('%s - %s\n%s', debug_message, description, reply)
        plugins = PluginContainer(self.options.plugins)
        ctx = plugins.message.received(reply=reply)
        reply = ctx.reply
        replyroot = None
        if status in (http.client.OK, http.client.INTERNAL_SERVER_ERROR):
            replyroot = _parse(reply)
            if len(reply) > 0:
                self.last_received(replyroot)
            plugins.message.parsed(reply=replyroot)
            fault = self.__get_fault(replyroot)
            if fault:
                if status != http.client.INTERNAL_SERVER_ERROR:
                    log.warning('Web service reported a SOAP processing fault using an unexpected HTTP status code %d. Reporting as an internal server error.', status)
                if self.options.faults:
                    raise WebFault(fault, replyroot)
                return (http.client.INTERNAL_SERVER_ERROR, fault)
        if status != http.client.OK:
            if self.options.faults:
                raise Exception((status, description))
            return (status, description)
        if self.options.retxml:
            return reply
        result = replyroot and self.method.binding.output.get_reply(self.method, replyroot)
        ctx = plugins.message.unmarshalled(reply=result)
        result = ctx.reply
        if self.options.faults:
            return result
        return (http.client.OK, result)

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

    def __location(self):
        """Returns the SOAP request's target location URL."""
        return Unskin(self.options).get('location', self.method.location)

    def last_sent(self, d=None):
        """
        Get or set last SOAP sent messages document

        To get the last sent document call the function without parameter.
        To set the last sent message, pass the document as parameter.

        @param d: A SOAP reply dict message key
        @type string: I{bytes}
        @return: The last sent I{soap} message.
        @rtype: L{Document}

        """
        key = 'tx'
        messages = self.client.messages
        if d is None:
            return messages.get(key)
        else:
            messages[key] = d

    def last_received(self, d=None):
        """
        Get or set last SOAP received messages document

        To get the last received document call the function without parameter.
        To set the last sent message, pass the document as parameter.

        @param d: A SOAP reply dict message key
        @type string: I{bytes}
        @return: The last received I{soap} message.
        @rtype: L{Document}

        """
        key = 'rx'
        messages = self.client.messages
        if d is None:
            return messages.get(key)
        else:
            messages[key] = d