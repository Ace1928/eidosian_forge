from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Construct(object):
    """
    The mother of all constructs.

    This object is generally not directly instantiated, and it does not
    directly implement parsing and building, so it is largely only of interest
    to subclass implementors.

    The external user API:

     * parse()
     * parse_stream()
     * build()
     * build_stream()
     * sizeof()

    Subclass authors should not override the external methods. Instead,
    another API is available:

     * _parse()
     * _build()
     * _sizeof()

    There is also a flag API:

     * _set_flag()
     * _clear_flag()
     * _inherit_flags()
     * _is_flag()

    And stateful copying:

     * __getstate__()
     * __setstate__()

    Attributes and Inheritance
    ==========================

    All constructs have a name and flags. The name is used for naming struct
    members and context dictionaries. Note that the name can either be a
    string, or None if the name is not needed. A single underscore ("_") is a
    reserved name, and so are names starting with a less-than character ("<").
    The name should be descriptive, short, and valid as a Python identifier,
    although these rules are not enforced.

    The flags specify additional behavioral information about this construct.
    Flags are used by enclosing constructs to determine a proper course of
    action. Flags are inherited by default, from inner subconstructs to outer
    constructs. The enclosing construct may set new flags or clear existing
    ones, as necessary.

    For example, if FLAG_COPY_CONTEXT is set, repeaters will pass a copy of
    the context for each iteration, which is necessary for OnDemand parsing.
    """
    FLAG_COPY_CONTEXT = 1
    FLAG_DYNAMIC = 2
    FLAG_EMBED = 4
    FLAG_NESTING = 8
    __slots__ = ['name', 'conflags']

    def __init__(self, name, flags=0):
        if name is not None:
            if type(name) is not str:
                raise TypeError('name must be a string or None', name)
            if name == '_' or name.startswith('<'):
                raise ValueError('reserved name', name)
        self.name = name
        self.conflags = flags

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.name)

    def _set_flag(self, flag):
        """
        Set the given flag or flags.

        :param int flag: flag to set; may be OR'd combination of flags
        """
        self.conflags |= flag

    def _clear_flag(self, flag):
        """
        Clear the given flag or flags.

        :param int flag: flag to clear; may be OR'd combination of flags
        """
        self.conflags &= ~flag

    def _inherit_flags(self, *subcons):
        """
        Pull flags from subconstructs.
        """
        for sc in subcons:
            self._set_flag(sc.conflags)

    def _is_flag(self, flag):
        """
        Check whether a given flag is set.

        :param int flag: flag to check
        """
        return bool(self.conflags & flag)

    def __getstate__(self):
        """
        Obtain a dictionary representing this construct's state.
        """
        attrs = {}
        if hasattr(self, '__dict__'):
            attrs.update(self.__dict__)
        slots = []
        c = self.__class__
        while c is not None:
            if hasattr(c, '__slots__'):
                slots.extend(c.__slots__)
            c = c.__base__
        for name in slots:
            if hasattr(self, name):
                attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, attrs):
        """
        Set this construct's state to a given state.
        """
        for name, value in attrs.items():
            setattr(self, name, value)

    def __copy__(self):
        """returns a copy of this construct"""
        self2 = object.__new__(self.__class__)
        self2.__setstate__(self.__getstate__())
        return self2

    def parse(self, data):
        """
        Parse an in-memory buffer.

        Strings, buffers, memoryviews, and other complete buffers can be
        parsed with this method.
        """
        return self.parse_stream(BytesIO(data))

    def parse_stream(self, stream):
        """
        Parse a stream.

        Files, pipes, sockets, and other streaming sources of data are handled
        by this method.
        """
        return self._parse(stream, Container())

    def _parse(self, stream, context):
        """
        Override me in your subclass.
        """
        raise NotImplementedError()

    def build(self, obj):
        """
        Build an object in memory.
        """
        stream = BytesIO()
        self.build_stream(obj, stream)
        return stream.getvalue()

    def build_stream(self, obj, stream):
        """
        Build an object directly into a stream.
        """
        self._build(obj, stream, Container())

    def _build(self, obj, stream, context):
        """
        Override me in your subclass.
        """
        raise NotImplementedError()

    def sizeof(self, context=None):
        """
        Calculate the size of this object, optionally using a context.

        Some constructs have no fixed size and can only know their size for a
        given hunk of data; these constructs will raise an error if they are
        not passed a context.

        :param ``Container`` context: contextual data

        :returns: int of the length of this construct
        :raises SizeofError: the size could not be determined
        """
        if context is None:
            context = Container()
        try:
            return self._sizeof(context)
        except Exception as e:
            raise SizeofError(e)

    def _sizeof(self, context):
        """
        Override me in your subclass.
        """
        raise SizeofError('Raw Constructs have no size!')