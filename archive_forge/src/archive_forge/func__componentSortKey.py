from pyasn1 import error
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import str2octs, null
from pyasn1.type import univ
from pyasn1.type import useful
@staticmethod
def _componentSortKey(componentAndType):
    """Sort SET components by tag

        Sort regardless of the Choice value (static sort)
        """
    component, asn1Spec = componentAndType
    if asn1Spec is None:
        asn1Spec = component
    if asn1Spec.typeId == univ.Choice.typeId and (not asn1Spec.tagSet):
        if asn1Spec.tagSet:
            return asn1Spec.tagSet
        else:
            return asn1Spec.componentType.minTagSet
    else:
        return asn1Spec.tagSet