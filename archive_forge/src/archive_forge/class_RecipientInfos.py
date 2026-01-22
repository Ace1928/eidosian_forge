from pyasn1_modules.rfc2459 import *
class RecipientInfos(univ.SetOf):
    componentType = RecipientInfo()