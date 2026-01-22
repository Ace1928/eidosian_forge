from pyasn1_modules.rfc2459 import *
class CertificateRevocationLists(univ.SetOf):
    componentType = CertificateRevocationList()