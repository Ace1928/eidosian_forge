from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class SubsequentMessage(univ.Integer):
    namedValues = namedval.NamedValues(('encrCert', 0), ('challengeResp', 1))