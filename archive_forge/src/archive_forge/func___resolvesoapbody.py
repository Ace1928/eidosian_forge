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
def __resolvesoapbody(self, definitions, op):
    """
        Resolve SOAP body I{message} parts by cross-referencing with operation
        defined in port type.

        @param definitions: A definitions object.
        @type definitions: L{Definitions}
        @param op: An I{operation} object.
        @type op: I{operation}

        """
    ptop = self.type.operation(op.name)
    if ptop is None:
        raise Exception("operation '%s' not defined in portType" % (op.name,))
    soap = op.soap
    parts = soap.input.body.parts
    if parts:
        pts = []
        for p in ptop.input.parts:
            if p.name in parts:
                pts.append(p)
        soap.input.body.parts = pts
    else:
        soap.input.body.parts = ptop.input.parts
    parts = soap.output.body.parts
    if parts:
        pts = []
        for p in ptop.output.parts:
            if p.name in parts:
                pts.append(p)
        soap.output.body.parts = pts
    else:
        soap.output.body.parts = ptop.output.parts