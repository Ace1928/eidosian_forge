from twisted.python import usage
from twisted.trial import unittest
class WrongTypedOptions(usage.Options):
    optParameters = [['barwrong', None, None, 'Bar with wrong coerce', 'he']]