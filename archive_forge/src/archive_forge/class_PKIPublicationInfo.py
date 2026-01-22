from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class PKIPublicationInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('action', univ.Integer(namedValues=namedval.NamedValues(('dontPublish', 0), ('pleasePublish', 1)))), namedtype.OptionalNamedType('pubInfos', univ.SequenceOf(componentType=SinglePubInfo()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))))