import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def unjellyAttribute(self, instance, attrName, ao):
    """Utility method for unjellying into instances of attributes.

        Use this rather than unjellyAO unless you like surprising bugs!
        Alternatively, you can use unjellyInto on your instance's __dict__.
        """
    self.unjellyInto(instance.__dict__, attrName, ao)