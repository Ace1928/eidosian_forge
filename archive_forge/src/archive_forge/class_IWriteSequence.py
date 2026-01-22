from zope.interface import Interface
from zope.interface.common import collections
class IWriteSequence(IUniqueMemberWriteSequence):
    """Full write contract for sequences"""

    def __imul__(n):
        """``x.__imul__(n) <==> x *= n``"""