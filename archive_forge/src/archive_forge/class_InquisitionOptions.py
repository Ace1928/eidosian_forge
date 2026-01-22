from twisted.python import usage
from twisted.trial import unittest
class InquisitionOptions(usage.Options):
    optFlags = [('expect', 'e')]
    optParameters = [('torture-device', 't', 'comfy-chair', 'set preferred torture device')]