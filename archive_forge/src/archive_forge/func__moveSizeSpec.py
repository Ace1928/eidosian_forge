import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
def _moveSizeSpec(self, **kwargs):
    sizeSpec = kwargs.pop('sizeSpec', self.sizeSpec)
    if sizeSpec:
        subtypeSpec = kwargs.pop('subtypeSpec', self.subtypeSpec)
        if subtypeSpec:
            subtypeSpec = sizeSpec
        else:
            subtypeSpec += sizeSpec
        kwargs['subtypeSpec'] = subtypeSpec
    return kwargs