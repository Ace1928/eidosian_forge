from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class CertReqMsg(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('certReq', CertRequest()), namedtype.OptionalNamedType('pop', ProofOfPossession()), namedtype.OptionalNamedType('regInfo', univ.SequenceOf(componentType=AttributeTypeAndValue()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))))