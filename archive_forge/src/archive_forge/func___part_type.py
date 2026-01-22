from suds import *
from suds.sax import Namespace
from suds.sax.document import Document
from suds.sax.element import Element
from suds.sudsobject import Factory
from suds.mx import Content
from suds.mx.literal import Literal as MxLiteral
from suds.umx.typed import Typed as UmxTyped
from suds.bindings.multiref import MultiRef
from suds.xsd.query import TypeQuery, ElementQuery
from suds.xsd.sxbasic import Element as SchemaElement
from suds.options import Options
from suds.plugin import PluginContainer
from copy import deepcopy
def __part_type(self, part, input):
    """
        Get a I{parameter definition} (pdef) defined for a given body or header
        message part.

        An input I{pdef} is a (I{name}, L{xsd.sxbase.SchemaObject}) tuple,
        while an output I{pdef} is a L{xsd.sxbase.SchemaObject}.

        @param part: A service method input or output part.
        @type part: I{suds.wsdl.Part}
        @param input: Defines input/output message.
        @type input: boolean
        @return:  A list of parameter definitions
        @rtype: [I{pdef},...]

        """
    if part.element is None:
        query = TypeQuery(part.type)
    else:
        query = ElementQuery(part.element)
    part_type = query.execute(self.schema())
    if part_type is None:
        raise TypeNotFound(query.ref)
    if part.type is not None:
        part_type = PartElement(part.name, part_type)
    if not input:
        return part_type
    if part_type.name is None:
        return (part.name, part_type)
    return (part_type.name, part_type)