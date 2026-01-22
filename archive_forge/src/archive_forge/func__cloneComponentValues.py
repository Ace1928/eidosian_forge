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