from collections import OrderedDict
from pyasn1 import debug
from pyasn1 import error
from pyasn1.compat import _MISSING
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class AnyEncoder(AbstractItemEncoder):

    def encode(self, value, encodeFun, **options):
        return value.asOctets()