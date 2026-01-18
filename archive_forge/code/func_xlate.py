from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def xlate(self, type):
    """
        Get a (namespace) translated I{qualified} name for specified type.
        @param type: A schema type.
        @type type: I{suds.xsd.sxbasic.SchemaObject}
        @return: A translated I{qualified} name.
        @rtype: str
        """
    resolved = type.resolve()
    name = resolved.name
    if type.multi_occurrence():
        name += '[]'
    ns = resolved.namespace()
    if ns[1] == self.wsdl.tns[1]:
        return name
    prefix = self.getprefix(ns[1])
    return ':'.join((prefix, name))