class DTDHandler:
    """Handle DTD events.

    This interface specifies only those DTD events required for basic
    parsing (unparsed entities and attributes)."""

    def notationDecl(self, name, publicId, systemId):
        """Handle a notation declaration event."""

    def unparsedEntityDecl(self, name, publicId, systemId, ndata):
        """Handle an unparsed entity declaration event."""