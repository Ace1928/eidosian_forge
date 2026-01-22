from suds import *
from logging import getLogger
class MessagePlugin(Plugin):
    """Base class for suds I{SOAP message} plugins."""

    def marshalled(self, context):
        """
        Suds is about to send the specified SOAP envelope.

        Provides the plugin with the opportunity to inspect/modify the envelope
        Document before it is sent.

        @param context: The send context.
            The I{envelope} is the envelope document.
        @type context: L{MessageContext}

        """
        pass

    def sending(self, context):
        """
        Suds is about to send the specified SOAP envelope.

        Provides the plugin with the opportunity to inspect/modify the message
        text before it is sent.

        @param context: The send context.
            The I{envelope} is the envelope text.
        @type context: L{MessageContext}

        """
        pass

    def received(self, context):
        """
        Suds has received the specified reply.

        Provides the plugin with the opportunity to inspect/modify the received
        XML text before it is SAX parsed.

        @param context: The reply context.
            The I{reply} is the raw text.
        @type context: L{MessageContext}

        """
        pass

    def parsed(self, context):
        """
        Suds has SAX parsed the received reply.

        Provides the plugin with the opportunity to inspect/modify the SAX
        parsed DOM tree for the reply before it is unmarshalled.

        @param context: The reply context.
            The I{reply} is DOM tree.
        @type context: L{MessageContext}

        """
        pass

    def unmarshalled(self, context):
        """
        Suds has unmarshalled the received reply.

        Provides the plugin with the opportunity to inspect/modify the
        unmarshalled reply object before it is returned.

        @param context: The reply context.
            The I{reply} is unmarshalled suds object.
        @type context: L{MessageContext}

        """
        pass