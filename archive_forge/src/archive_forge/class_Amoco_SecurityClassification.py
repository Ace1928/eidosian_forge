from pyasn1.type import char
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5755
class Amoco_SecurityClassification(univ.Integer):
    namedValues = namedval.NamedValues(('amoco-general', 6), ('amoco-confidential', 7), ('amoco-highly-confidential', 8))