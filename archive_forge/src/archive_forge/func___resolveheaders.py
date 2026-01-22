from suds import *
from suds.bindings.document import Document
from suds.bindings.rpc import RPC, Encoded
from suds.reader import DocumentReader
from suds.sax.element import Element
from suds.sudsobject import Object, Facade, Metadata
from suds.xsd import qualify, Namespace
from suds.xsd.query import ElementQuery
from suds.xsd.schema import Schema, SchemaCollection
import re
from . import soaparray
from urllib.parse import urljoin
from logging import getLogger
def __resolveheaders(self, definitions, op):
    """
        Resolve SOAP header I{message} references.

        @param definitions: A definitions object.
        @type definitions: L{Definitions}
        @param op: An I{operation} object.
        @type op: I{operation}

        """
    soap = op.soap
    headers = soap.input.headers + soap.output.headers
    for header in headers:
        mn = header.message
        ref = qualify(mn, self.root, definitions.tns)
        message = definitions.messages.get(ref)
        if message is None:
            raise Exception("message '%s', not-found" % (mn,))
        pn = header.part
        for p in message.parts:
            if p.name == pn:
                header.part = p
                break
        if pn == header.part:
            raise Exception("message '%s' has not part named '%s'" % (ref, pn))