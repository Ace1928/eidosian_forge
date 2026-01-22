from pyasn1_modules import rfc2251
from pyasn1_modules.rfc2459 import *
class Attributes(univ.SetOf):
    componentType = rfc2251.Attribute()