from pyasn1_modules.rfc2459 import *
class SignerInfos(univ.SetOf):
    componentType = SignerInfo()