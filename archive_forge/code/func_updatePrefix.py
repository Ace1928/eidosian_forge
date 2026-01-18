from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def updatePrefix(self, p, u):
    """
        Update (redefine) a prefix mapping for the branch.

        @param p: A prefix.
        @type p: basestring
        @param u: A namespace URI.
        @type u: basestring
        @return: self
        @rtype: L{Element}
        @note: This method traverses down the entire branch!

        """
    if p in self.nsprefixes:
        self.nsprefixes[p] = u
    for c in self.children:
        c.updatePrefix(p, u)
    return self