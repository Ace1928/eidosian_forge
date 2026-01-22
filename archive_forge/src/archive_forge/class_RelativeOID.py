import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
class RelativeOID(base.SimpleAsn1Type):
    """Create |ASN.1| schema or value object.
    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`, its
    objects are immutable and duck-type Python :class:`tuple` objects
    (tuple of non-negative integers).
    Keyword Args
    ------------
    value: :class:`tuple`, :class:`str` or |ASN.1| object
        Python sequence of :class:`int` or :class:`str` literal or |ASN.1| object.
        If `value` is not given, schema object will be created.
    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)
    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type occurs automatically on object
        instantiation.
    Raises
    ------
    ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
        On constraint violation or bad initializer.
    Examples
    --------
    .. code-block:: python
        class RelOID(RelativeOID):
            '''
            ASN.1 specification:
            id-pad-null RELATIVE-OID ::= { 0 }
            id-pad-once RELATIVE-OID ::= { 5 6 }
            id-pad-twice RELATIVE-OID ::= { 5 6 7 }
            '''
        id_pad_null = RelOID('0')
        id_pad_once = RelOID('5.6')
        id_pad_twice = id_pad_once + (7,)
    """
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 13))
    subtypeSpec = constraint.ConstraintsIntersection()
    typeId = base.SimpleAsn1Type.getTypeId()

    def __add__(self, other):
        return self.clone(self._value + other)

    def __radd__(self, other):
        return self.clone(other + self._value)

    def asTuple(self):
        return self._value

    def __len__(self):
        return len(self._value)

    def __getitem__(self, i):
        if i.__class__ is slice:
            return self.clone(self._value[i])
        else:
            return self._value[i]

    def __iter__(self):
        return iter(self._value)

    def __contains__(self, value):
        return value in self._value

    def index(self, suboid):
        return self._value.index(suboid)

    def isPrefixOf(self, other):
        """Indicate if this |ASN.1| object is a prefix of other |ASN.1| object.
        Parameters
        ----------
        other: |ASN.1| object
            |ASN.1| object
        Returns
        -------
        : :class:`bool`
            :obj:`True` if this |ASN.1| object is a parent (e.g. prefix) of the other |ASN.1| object
            or :obj:`False` otherwise.
        """
        l = len(self)
        if l <= len(other):
            if self._value[:l] == other[:l]:
                return True
        return False

    def prettyIn(self, value):
        if isinstance(value, RelativeOID):
            return tuple(value)
        elif octets.isStringType(value):
            if '-' in value:
                raise error.PyAsn1Error('Malformed RELATIVE-OID %s at %s: %s' % (value, self.__class__.__name__, sys.exc_info()[1]))
            try:
                return tuple([int(subOid) for subOid in value.split('.') if subOid])
            except ValueError:
                raise error.PyAsn1Error('Malformed RELATIVE-OID %s at %s: %s' % (value, self.__class__.__name__, sys.exc_info()[1]))
        try:
            tupleOfInts = tuple([int(subOid) for subOid in value if subOid >= 0])
        except (ValueError, TypeError):
            raise error.PyAsn1Error('Malformed RELATIVE-OID %s at %s: %s' % (value, self.__class__.__name__, sys.exc_info()[1]))
        if len(tupleOfInts) == len(value):
            return tupleOfInts
        raise error.PyAsn1Error('Malformed RELATIVE-OID %s at %s' % (value, self.__class__.__name__))

    def prettyOut(self, value):
        return '.'.join([str(x) for x in value])