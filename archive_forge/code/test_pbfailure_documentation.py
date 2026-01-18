from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest

        When L{CopiedFailure.printTraceback} is used to print a copied failure
        which was unjellied from a L{CopyableFailure} with C{unsafeTracebacks}
        set to C{False}, the string representation of the exception value is
        included in the output.
        