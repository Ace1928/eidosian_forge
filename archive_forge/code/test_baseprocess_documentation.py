from twisted.internet._baseprocess import BaseProcess
from twisted.python.deprecate import getWarningMethod, setWarningMethod
from twisted.trial.unittest import TestCase

        L{BaseProcess._callProcessExited} emits a L{DeprecationWarning} if the
        object referred to by its C{proto} attribute has no C{processExited}
        method.
        