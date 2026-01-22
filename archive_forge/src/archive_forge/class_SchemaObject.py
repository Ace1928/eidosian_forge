from suds import *
from suds.xsd import *
from suds.sax.element import Element
from suds.sax import Namespace
from logging import getLogger
class SchemaObject(UnicodeMixin):
    """
    A schema object is an extension to object with schema awareness.

    @ivar root: The XML root element.
    @type root: L{Element}
    @ivar schema: The schema containing this object.
    @type schema: L{schema.Schema}
    @ivar form_qualified: A flag indicating that @elementFormDefault has a
        value of I{qualified}.
    @type form_qualified: boolean
    @ivar nillable: A flag indicating that @nillable has a value of I{true}.
    @type nillable: boolean
    @ivar default: The default value.
    @type default: object
    @ivar rawchildren: A list raw of all children.
    @type rawchildren: [L{SchemaObject},...]

    """

    @classmethod
    def prepend(cls, d, s, filter=Filter()):
        """
        Prepend B{s}ource XSD schema objects to the B{d}estination list.

        B{filter} is used to decide which objects to prepend and which to skip.

        @param d: The destination list.
        @type d: list
        @param s: The source list.
        @type s: list
        @param filter: A filter allowing items to be prepended.
        @type filter: L{Filter}

        """
        i = 0
        for x in s:
            if x in filter:
                d.insert(i, x)
                i += 1

    @classmethod
    def append(cls, d, s, filter=Filter()):
        """
        Append B{s}ource XSD schema objects to the B{d}estination list.

        B{filter} is used to decide which objects to append and which to skip.

        @param d: The destination list.
        @type d: list
        @param s: The source list.
        @type s: list
        @param filter: A filter that allows items to be appended.
        @type filter: L{Filter}

        """
        for item in s:
            if item in filter:
                d.append(item)

    def __init__(self, schema, root):
        """
        @param schema: The containing schema.
        @type schema: L{schema.Schema}
        @param root: The XML root node.
        @type root: L{Element}

        """
        self.schema = schema
        self.root = root
        self.id = objid(self)
        self.name = root.get('name')
        self.qname = (self.name, schema.tns[1])
        self.min = root.get('minOccurs')
        self.max = root.get('maxOccurs')
        self.type = root.get('type')
        self.ref = root.get('ref')
        self.form_qualified = schema.form_qualified
        self.nillable = False
        self.default = root.get('default')
        self.rawchildren = []

    def attributes(self, filter=Filter()):
        """
        Get only the attribute content.

        @param filter: A filter to constrain the result.
        @type filter: L{Filter}
        @return: A list of (attr, ancestry) tuples.
        @rtype: [(L{SchemaObject}, [L{SchemaObject},..]),..]

        """
        result = []
        for child, ancestry in self:
            if child.isattr() and child in filter:
                result.append((child, ancestry))
        return result

    def children(self, filter=Filter()):
        """
        Get only the I{direct} or non-attribute content.

        @param filter: A filter to constrain the result.
        @type filter: L{Filter}
        @return: A list tuples: (child, ancestry)
        @rtype: [(L{SchemaObject}, [L{SchemaObject},..]),..]

        """
        result = []
        for child, ancestry in self:
            if not child.isattr() and child in filter:
                result.append((child, ancestry))
        return result

    def get_attribute(self, name):
        """
        Get (find) an attribute by name.

        @param name: A attribute name.
        @type name: str
        @return: A tuple: the requested (attribute, ancestry).
        @rtype: (L{SchemaObject}, [L{SchemaObject},..])

        """
        for child, ancestry in self.attributes():
            if child.name == name:
                return (child, ancestry)
        return (None, [])

    def get_child(self, name):
        """
        Get (find) a I{non-attribute} child by name.

        @param name: A child name.
        @type name: str
        @return: A tuple: the requested (child, ancestry).
        @rtype: (L{SchemaObject}, [L{SchemaObject},..])

        """
        for child, ancestry in self.children():
            if child.any() or child.name == name:
                return (child, ancestry)
        return (None, [])

    def namespace(self, prefix=None):
        """
        Get this property's namespace.

        @param prefix: The default prefix.
        @type prefix: str
        @return: The schema's target namespace.
        @rtype: (I{prefix}, I{URI})

        """
        ns = self.schema.tns
        if ns[0] is None:
            ns = (prefix, ns[1])
        return ns

    def default_namespace(self):
        return self.root.defaultNamespace()

    def multi_occurrence(self):
        """
        Get whether the node has multiple occurrences, i.e. is a I{collection}.

        @return: True if it has, False if it has at most 1 occurrence.
        @rtype: boolean

        """
        max = self.max
        if max is None:
            return False
        if max.isdigit():
            return int(max) > 1
        return max == 'unbounded'

    def optional(self):
        """
        Get whether this type is optional.

        @return: True if optional, else False.
        @rtype: boolean

        """
        return self.min == '0'

    def required(self):
        """
        Get whether this type is required.

        @return: True if required, else False.
        @rtype: boolean

        """
        return not self.optional()

    def resolve(self, nobuiltin=False):
        """
        Resolve the node's type reference and return the referenced type node.

        Only XSD schema objects that actually support 'having a type' custom
        implement this interface while others simply resolve as themselves.

        @param nobuiltin: Flag indicating whether resolving to an external XSD
            built-in type should not be allowed.
        @return: The resolved (true) type.
        @rtype: L{SchemaObject}
        """
        return self

    def sequence(self):
        """
        Get whether this is an <xsd:sequence/>.

        @return: True if <xsd:sequence/>, else False.
        @rtype: boolean

        """
        return False

    def xslist(self):
        """
        Get whether this is an <xsd:list/>.

        @return: True if <xsd:list/>, else False.
        @rtype: boolean

        """
        return False

    def all(self):
        """
        Get whether this is an <xsd:all/>.

        @return: True if <xsd:all/>, else False.
        @rtype: boolean

        """
        return False

    def choice(self):
        """
        Get whether this is an <xsd:choice/>.

        @return: True if <xsd:choice/>, else False.
        @rtype: boolean

        """
        return False

    def any(self):
        """
        Get whether this is an <xsd:any/>.

        @return: True if <xsd:any/>, else False.
        @rtype: boolean

        """
        return False

    def builtin(self):
        """
        Get whether this is a built-in schema-instance XSD type.

        @return: True if a built-in type, else False.
        @rtype: boolean

        """
        return False

    def enum(self):
        """
        Get whether this is a simple-type containing an enumeration.

        @return: True if enumeration, else False.
        @rtype: boolean

        """
        return False

    def isattr(self):
        """
        Get whether the object is a schema I{attribute} definition.

        @return: True if an attribute, else False.
        @rtype: boolean

        """
        return False

    def extension(self):
        """
        Get whether the object is an extension of another type.

        @return: True if an extension, else False.
        @rtype: boolean

        """
        return False

    def restriction(self):
        """
        Get whether the object is an restriction of another type.

        @return: True if a restriction, else False.
        @rtype: boolean

        """
        return False

    def mixed(self):
        """Get whether the object has I{mixed} content."""
        return False

    def find(self, qref, classes=[], ignore=None):
        """
        Find a referenced type in self or children. Return None if not found.

        Qualified references for all schema objects checked in this search will
        be added to the set of ignored qualified references to avoid the find
        operation going into an infinite loop in case of recursively defined
        structures.

        @param qref: A qualified reference.
        @type qref: qref
        @param classes: A collection of classes used to qualify the match.
        @type classes: Collection(I{class},...), e.g. [I(class),...]
        @param ignore: A set of qualified references to ignore in this search.
        @type ignore: {qref,...}
        @return: The referenced type.
        @rtype: L{SchemaObject}
        @see: L{qualify()}

        """
        if not len(classes):
            classes = (self.__class__,)
        if ignore is None:
            ignore = set()
        if self.qname in ignore:
            return
        ignore.add(self.qname)
        if self.qname == qref and self.__class__ in classes:
            return self
        for c in self.rawchildren:
            p = c.find(qref, classes, ignore=ignore)
            if p is not None:
                return p

    def translate(self, value, topython=True):
        """
        Translate between an XSD type values and Python objects.

        When converting a Python object to an XSD type value the operation may
        return any Python object whose string representation matches the
        desired XSD type value.

        @param value: A value to translate.
        @type value: str if topython is True; any Python object otherwise
        @param topython: Flag indicating the translation direction.
        @type topython: bool
        @return: The converted I{language} type.

        """
        return value

    def childtags(self):
        """
        Get a list of valid child tag names.

        @return: A list of child tag names.
        @rtype: [str,...]

        """
        return ()

    def dependencies(self):
        """
        Get a list of dependencies for dereferencing.

        @return: A merge dependency index and a list of dependencies.
        @rtype: (int, [L{SchemaObject},...])

        """
        return (None, [])

    def autoqualified(self):
        """
        The list of I{auto} qualified attribute values.

        Qualification means to convert values into I{qref}.

        @return: A list of attribute names.
        @rtype: list

        """
        return ['type', 'ref']

    def qualify(self):
        """
        Convert reference attribute values into a I{qref}.

        Constructed I{qref} uses the default document namespace. Since many
        WSDL schemas are written improperly: when the document does not define
        its default namespace, the schema target namespace is used to qualify
        references.

        """
        defns = self.root.defaultNamespace()
        if Namespace.none(defns):
            defns = self.schema.tns
        for a in self.autoqualified():
            ref = getattr(self, a)
            if ref is None:
                continue
            if isqref(ref):
                continue
            qref = qualify(ref, self.root, defns)
            log.debug("%s, convert %s='%s' to %s", self.id, a, ref, qref)
            setattr(self, a, qref)

    def merge(self, other):
        """Merge another object as needed."""
        other.qualify()
        for n in ('default', 'max', 'min', 'name', 'nillable', 'qname', 'type'):
            if getattr(self, n) is not None:
                continue
            v = getattr(other, n)
            if v is None:
                continue
            setattr(self, n, v)

    def content(self, collection=None, filter=Filter(), history=None):
        """
        Get a I{flattened} list of this node's contents.

        @param collection: A list to fill.
        @type collection: list
        @param filter: A filter used to constrain the result.
        @type filter: L{Filter}
        @param history: The history list used to prevent cyclic dependency.
        @type history: list
        @return: The filled list.
        @rtype: list

        """
        if collection is None:
            collection = []
        if history is None:
            history = []
        if self in history:
            return collection
        history.append(self)
        if self in filter:
            collection.append(self)
        for c in self.rawchildren:
            c.content(collection, filter, history)
        history.pop()
        return collection

    def str(self, indent=0, history=None):
        """
        Get a string representation of this object.

        @param indent: The indent.
        @type indent: int
        @return: A string.
        @rtype: str

        """
        if history is None:
            history = []
        if self in history:
            return '%s ...' % Repr(self)
        history.append(self)
        tab = '%*s' % (indent * 3, '')
        result = ['%s<%s' % (tab, self.id)]
        for n in self.description():
            if not hasattr(self, n):
                continue
            v = getattr(self, n)
            if v is None:
                continue
            result.append(' %s="%s"' % (n, v))
        if len(self):
            result.append('>')
            for c in self.rawchildren:
                result.append('\n')
                result.append(c.str(indent + 1, history[:]))
                if c.isattr():
                    result.append('@')
            result.append('\n%s' % (tab,))
            result.append('</%s>' % (self.__class__.__name__,))
        else:
            result.append(' />')
        return ''.join(result)

    def description(self):
        """
        Get the names used for repr() and str() description.

        @return:  A dictionary of relevant attributes.
        @rtype: [str,...]

        """
        return ()

    def __unicode__(self):
        return str(self.str())

    def __repr__(self):
        s = []
        s.append('<%s' % (self.id,))
        for n in self.description():
            if not hasattr(self, n):
                continue
            v = getattr(self, n)
            if v is None:
                continue
            s.append(' %s="%s"' % (n, v))
        s.append(' />')
        return ''.join(s)

    def __len__(self):
        n = 0
        for x in self:
            n += 1
        return n

    def __iter__(self):
        return Iter(self)

    def __getitem__(self, index):
        """
        Returns a contained schema object referenced by its 0-based index.

        Returns None if such an object does not exist.

        """
        i = 0
        for c in self:
            if i == index:
                return c
            i += 1