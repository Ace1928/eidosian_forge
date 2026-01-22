from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
class siren_dn(SIREntityName):

    def __init__(self):
        SIREntityName.__init__(self)
        self['sirenType'] = id_dn