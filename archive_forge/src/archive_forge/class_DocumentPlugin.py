from suds import *
from logging import getLogger
class DocumentPlugin(Plugin):
    """Base class for suds I{document} plugins."""

    def loaded(self, context):
        """
        Suds has loaded a WSDL/XSD document.

        Provides the plugin with an opportunity to inspect/modify the unparsed
        document. Called after each WSDL/XSD document is loaded.

        @param context: The document context.
        @type context: L{DocumentContext}

        """
        pass

    def parsed(self, context):
        """
        Suds has parsed a WSDL/XSD document.

        Provides the plugin with an opportunity to inspect/modify the parsed
        document. Called after each WSDL/XSD document is parsed.

        @param context: The document context.
        @type context: L{DocumentContext}

        """
        pass