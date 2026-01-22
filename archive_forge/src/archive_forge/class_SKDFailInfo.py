from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc3565
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc5755
class SKDFailInfo(univ.Integer):
    namedValues = namedval.NamedValues(('unspecified', 0), ('closedGL', 1), ('unsupportedDuration', 2), ('noGLACertificate', 3), ('invalidCert', 4), ('unsupportedAlgorithm', 5), ('noGLONameMatch', 6), ('invalidGLName', 7), ('nameAlreadyInUse', 8), ('noSpam', 9), ('alreadyAMember', 11), ('notAMember', 12), ('alreadyAnOwner', 13), ('notAnOwner', 14))