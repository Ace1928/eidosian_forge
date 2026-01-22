from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class ECPVer(univ.Integer):
    namedValues = namedval.NamedValues(('ecpVer1', 1))