from suds import *
from suds.sax import splitPrefix, Namespace
from suds.sudsobject import Object
from suds.xsd.query import BlindQuery, TypeQuery, qualify
import re
from logging import getLogger
class PathResolver(Resolver):
    """
    Resolves the definition object for the schema type located at a given path.
    The path may contain (.) dot notation to specify nested types.
    @ivar wsdl: A wsdl object.
    @type wsdl: L{wsdl.Definitions}
    """

    def __init__(self, wsdl, ps='.'):
        """
        @param wsdl: A schema object.
        @type wsdl: L{wsdl.Definitions}
        @param ps: The path separator character
        @type ps: char
        """
        Resolver.__init__(self, wsdl.schema)
        self.wsdl = wsdl
        self.altp = re.compile('({)(.+)(})(.+)')
        self.splitp = re.compile('({.+})*[^\\%s]+' % ps[0])

    def find(self, path, resolved=True):
        """
        Get the definition object for the schema type located at the specified path.
        The path may contain (.) dot notation to specify nested types.
        Actually, the path separator is usually a (.) but can be redefined
        during contruction.
        @param path: A (.) separated path to a schema type.
        @type path: basestring
        @param resolved: A flag indicating that the fully resolved type
            should be returned.
        @type resolved: boolean
        @return: The found schema I{type}
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        result = None
        parts = self.split(path)
        try:
            result = self.root(parts)
            if len(parts) > 1:
                result = result.resolve(nobuiltin=True)
                result = self.branch(result, parts)
                result = self.leaf(result, parts)
            if resolved:
                result = result.resolve(nobuiltin=True)
        except PathResolver.BadPath:
            log.error('path: "%s", not-found' % path)
        return result

    def root(self, parts):
        """
        Find the path root.
        @param parts: A list of path parts.
        @type parts: [str,..]
        @return: The root.
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        result = None
        name = parts[0]
        log.debug('searching schema for (%s)', name)
        qref = self.qualify(parts[0])
        query = BlindQuery(qref)
        result = query.execute(self.schema)
        if result is None:
            log.error('(%s) not-found', name)
            raise PathResolver.BadPath(name)
        log.debug('found (%s) as (%s)', name, Repr(result))
        return result

    def branch(self, root, parts):
        """
        Traverse the path until a leaf is reached.
        @param parts: A list of path parts.
        @type parts: [str,..]
        @param root: The root.
        @type root: L{xsd.sxbase.SchemaObject}
        @return: The end of the branch.
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        result = root
        for part in parts[1:-1]:
            name = splitPrefix(part)[1]
            log.debug('searching parent (%s) for (%s)', Repr(result), name)
            result, ancestry = result.get_child(name)
            if result is None:
                log.error('(%s) not-found', name)
                raise PathResolver.BadPath(name)
            result = result.resolve(nobuiltin=True)
            log.debug('found (%s) as (%s)', name, Repr(result))
        return result

    def leaf(self, parent, parts):
        """
        Find the leaf.
        @param parts: A list of path parts.
        @type parts: [str,..]
        @param parent: The leaf's parent.
        @type parent: L{xsd.sxbase.SchemaObject}
        @return: The leaf.
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        name = splitPrefix(parts[-1])[1]
        if name.startswith('@'):
            result, path = parent.get_attribute(name[1:])
        else:
            result, ancestry = parent.get_child(name)
        if result is None:
            raise PathResolver.BadPath(name)
        return result

    def qualify(self, name):
        """
        Qualify the name as either:
          - plain name
          - ns prefixed name (eg: ns0:Person)
          - fully ns qualified name (eg: {http://myns-uri}Person)
        @param name: The name of an object in the schema.
        @type name: str
        @return: A qualified name.
        @rtype: qname
        """
        m = self.altp.match(name)
        if m is None:
            return qualify(name, self.wsdl.root, self.wsdl.tns)
        else:
            return (m.group(4), m.group(2))

    def split(self, s):
        """
        Split the string on (.) while preserving any (.) inside the
        '{}' alternalte syntax for full ns qualification.
        @param s: A plain or qualified name.
        @type s: str
        @return: A list of the name's parts.
        @rtype: [str,..]
        """
        parts = []
        b = 0
        while 1:
            m = self.splitp.match(s, b)
            if m is None:
                break
            b, e = m.span()
            parts.append(s[b:e])
            b = e + 1
        return parts

    class BadPath(Exception):
        pass