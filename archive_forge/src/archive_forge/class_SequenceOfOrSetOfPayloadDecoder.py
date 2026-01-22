from pyasn1 import debug
from pyasn1 import error
from pyasn1.compat import _MISSING
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class SequenceOfOrSetOfPayloadDecoder(object):

    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        asn1Value = asn1Spec.clone()
        for pyValue in pyObject:
            asn1Value.append(decodeFun(pyValue, asn1Spec.componentType), **options)
        return asn1Value