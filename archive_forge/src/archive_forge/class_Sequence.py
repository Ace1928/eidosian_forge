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
class Sequence(SequenceAndSetBase):
    __doc__ = SequenceAndSetBase.__doc__
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatConstructed, 16))
    subtypeSpec = constraint.ConstraintsIntersection()
    componentType = namedtype.NamedTypes()
    typeId = SequenceAndSetBase.getTypeId()

    def getComponentTagMapNearPosition(self, idx):
        if self.componentType:
            return self.componentType.getTagMapNearPosition(idx)

    def getComponentPositionNearType(self, tagSet, idx):
        if self.componentType:
            return self.componentType.getPositionNearType(tagSet, idx)
        else:
            return idx