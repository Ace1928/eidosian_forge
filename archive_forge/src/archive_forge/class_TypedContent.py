from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
class TypedContent(Content):
    """Represents any I{typed} content."""

    def __init__(self, *args, **kwargs):
        Content.__init__(self, *args, **kwargs)
        self.resolved_cache = {}

    def resolve(self, nobuiltin=False):
        """
        Resolve the node's type reference and return the referenced type node.

        Returns self if the type is defined locally, e.g. as a <complexType>
        subnode. Otherwise returns the referenced external node.

        @param nobuiltin: Flag indicating whether resolving to XSD built-in
            types should not be allowed.
        @return: The resolved (true) type.
        @rtype: L{SchemaObject}

        """
        cached = self.resolved_cache.get(nobuiltin)
        if cached is not None:
            return cached
        resolved = self.__resolve_type(nobuiltin)
        self.resolved_cache[nobuiltin] = resolved
        return resolved

    def __resolve_type(self, nobuiltin=False):
        """
        Private resolve() worker without any result caching.

        @param nobuiltin: Flag indicating whether resolving to XSD built-in
            types should not be allowed.
        @return: The resolved (true) type.
        @rtype: L{SchemaObject}

        """
        qref = self.qref()
        if qref is None:
            return self
        query = TypeQuery(qref)
        query.history = [self]
        log.debug('%s, resolving: %s\n using:%s', self.id, qref, query)
        resolved = query.execute(self.schema)
        if resolved is None:
            log.debug(self.schema)
            raise TypeNotFound(qref)
        if resolved.builtin() and nobuiltin:
            return self
        return resolved

    def qref(self):
        """
        Get the I{type} qualified reference to the referenced XSD type.

        This method takes into account simple types defined through restriction
        which are detected by determining that self is simple (len == 0) and by
        finding a restriction child.

        @return: The I{type} qualified reference.
        @rtype: qref

        """
        qref = self.type
        if qref is None and len(self) == 0:
            ls = []
            m = RestrictionMatcher()
            finder = NodeFinder(m, 1)
            finder.find(self, ls)
            if ls:
                return ls[0].ref
        return qref