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
class Choice(Set):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.ConstructedAsn1Type`,
    its objects are mutable and duck-type Python :class:`list` objects.

    Keyword Args
    ------------
    componentType: :py:class:`~pyasn1.type.namedtype.NamedType`
        Object holding named ASN.1 types allowed within this collection

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s).  Constraints
        verification for |ASN.1| type can only occur on explicit
        `.isInconsistent` call.

    Examples
    --------

    .. code-block:: python

        class Afters(Choice):
            '''
            ASN.1 specification:

            Afters ::= CHOICE {
                cheese  [0] IA5String,
                dessert [1] IA5String
            }
            '''
            componentType = NamedTypes(
                NamedType('cheese', IA5String().subtype(
                    implicitTag=Tag(tagClassContext, tagFormatSimple, 0)
                ),
                NamedType('dessert', IA5String().subtype(
                    implicitTag=Tag(tagClassContext, tagFormatSimple, 1)
                )
            )

        afters = Afters()
        afters['cheese'] = 'Mascarpone'
    """
    tagSet = tag.TagSet()
    componentType = namedtype.NamedTypes()
    subtypeSpec = constraint.ConstraintsIntersection(constraint.ValueSizeConstraint(1, 1))
    typeId = Set.getTypeId()
    _currentIdx = None

    def __eq__(self, other):
        if self._componentValues:
            return self._componentValues[self._currentIdx] == other
        return NotImplemented

    def __ne__(self, other):
        if self._componentValues:
            return self._componentValues[self._currentIdx] != other
        return NotImplemented

    def __lt__(self, other):
        if self._componentValues:
            return self._componentValues[self._currentIdx] < other
        return NotImplemented

    def __le__(self, other):
        if self._componentValues:
            return self._componentValues[self._currentIdx] <= other
        return NotImplemented

    def __gt__(self, other):
        if self._componentValues:
            return self._componentValues[self._currentIdx] > other
        return NotImplemented

    def __ge__(self, other):
        if self._componentValues:
            return self._componentValues[self._currentIdx] >= other
        return NotImplemented
    if sys.version_info[0] <= 2:

        def __nonzero__(self):
            return self._componentValues and True or False
    else:

        def __bool__(self):
            return self._componentValues and True or False

    def __len__(self):
        return self._currentIdx is not None and 1 or 0

    def __contains__(self, key):
        if self._currentIdx is None:
            return False
        return key == self.componentType[self._currentIdx].getName()

    def __iter__(self):
        if self._currentIdx is None:
            raise StopIteration
        yield self.componentType[self._currentIdx].getName()

    def values(self):
        if self._currentIdx is not None:
            yield self._componentValues[self._currentIdx]

    def keys(self):
        if self._currentIdx is not None:
            yield self.componentType[self._currentIdx].getName()

    def items(self):
        if self._currentIdx is not None:
            yield (self.componentType[self._currentIdx].getName(), self[self._currentIdx])

    def checkConsistency(self):
        if self._currentIdx is None:
            raise error.PyAsn1Error('Component not chosen')

    def _cloneComponentValues(self, myClone, cloneValueFlag):
        try:
            component = self.getComponent()
        except error.PyAsn1Error:
            pass
        else:
            if isinstance(component, Choice):
                tagSet = component.effectiveTagSet
            else:
                tagSet = component.tagSet
            if isinstance(component, base.ConstructedAsn1Type):
                myClone.setComponentByType(tagSet, component.clone(cloneValueFlag=cloneValueFlag))
            else:
                myClone.setComponentByType(tagSet, component.clone())

    def getComponentByPosition(self, idx, default=noValue, instantiate=True):
        __doc__ = Set.__doc__
        if self._currentIdx is None or self._currentIdx != idx:
            return Set.getComponentByPosition(self, idx, default=default, instantiate=instantiate)
        return self._componentValues[idx]

    def setComponentByPosition(self, idx, value=noValue, verifyConstraints=True, matchTags=True, matchConstraints=True):
        """Assign |ASN.1| type component by position.

        Equivalent to Python sequence item assignment operation (e.g. `[]`).

        Parameters
        ----------
        idx: :class:`int`
            Component index (zero-based). Must either refer to existing
            component or to N+1 component. In the latter case a new component
            type gets instantiated (if *componentType* is set, or given ASN.1
            object is taken otherwise) and appended to the |ASN.1| sequence.

        Keyword Args
        ------------
        value: :class:`object` or :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
            A Python value to initialize |ASN.1| component with (if *componentType* is set)
            or ASN.1 value object to assign to |ASN.1| component. Once a new value is
            set to *idx* component, previous value is dropped.
            If `value` is not given, schema object will be set as a component.

        verifyConstraints : :class:`bool`
            If :obj:`False`, skip constraints validation

        matchTags: :class:`bool`
            If :obj:`False`, skip component tags matching

        matchConstraints: :class:`bool`
            If :obj:`False`, skip component constraints matching

        Returns
        -------
        self
        """
        oldIdx = self._currentIdx
        Set.setComponentByPosition(self, idx, value, verifyConstraints, matchTags, matchConstraints)
        self._currentIdx = idx
        if oldIdx is not None and oldIdx != idx:
            self._componentValues[oldIdx] = noValue
        return self

    @property
    def effectiveTagSet(self):
        """Return a :class:`~pyasn1.type.tag.TagSet` object of the currently initialized component or self (if |ASN.1| is tagged)."""
        if self.tagSet:
            return self.tagSet
        else:
            component = self.getComponent()
            return component.effectiveTagSet

    @property
    def tagMap(self):
        """"Return a :class:`~pyasn1.type.tagmap.TagMap` object mapping
            ASN.1 tags to ASN.1 objects contained within callee.
        """
        if self.tagSet:
            return Set.tagMap.fget(self)
        else:
            return self.componentType.tagMapUnique

    def getComponent(self, innerFlag=False):
        """Return currently assigned component of the |ASN.1| object.

        Returns
        -------
        : :py:class:`~pyasn1.type.base.PyAsn1Item`
            a PyASN1 object
        """
        if self._currentIdx is None:
            raise error.PyAsn1Error('Component not chosen')
        else:
            c = self._componentValues[self._currentIdx]
            if innerFlag and isinstance(c, Choice):
                return c.getComponent(innerFlag)
            else:
                return c

    def getName(self, innerFlag=False):
        """Return the name of currently assigned component of the |ASN.1| object.

        Returns
        -------
        : :py:class:`str`
            |ASN.1| component name
        """
        if self._currentIdx is None:
            raise error.PyAsn1Error('Component not chosen')
        else:
            if innerFlag:
                c = self._componentValues[self._currentIdx]
                if isinstance(c, Choice):
                    return c.getName(innerFlag)
            return self.componentType.getNameByPosition(self._currentIdx)

    @property
    def isValue(self):
        """Indicate that |ASN.1| object represents ASN.1 value.

        If *isValue* is :obj:`False` then this object represents just ASN.1 schema.

        If *isValue* is :obj:`True` then, in addition to its ASN.1 schema features,
        this object can also be used like a Python built-in object (e.g.
        :class:`int`, :class:`str`, :class:`dict` etc.).

        Returns
        -------
        : :class:`bool`
            :obj:`False` if object represents just ASN.1 schema.
            :obj:`True` if object represents ASN.1 schema and can be used as a normal
            value.

        Note
        ----
        There is an important distinction between PyASN1 schema and value objects.
        The PyASN1 schema objects can only participate in ASN.1 schema-related
        operations (e.g. defining or testing the structure of the data). Most
        obvious uses of ASN.1 schema is to guide serialisation codecs whilst
        encoding/decoding serialised ASN.1 contents.

        The PyASN1 value objects can **additionally** participate in many operations
        involving regular Python objects (e.g. arithmetic, comprehension etc).
        """
        if self._currentIdx is None:
            return False
        componentValue = self._componentValues[self._currentIdx]
        return componentValue is not noValue and componentValue.isValue

    def clear(self):
        self._currentIdx = None
        return Set.clear(self)

    def getMinTagSet(self):
        return self.minTagSet