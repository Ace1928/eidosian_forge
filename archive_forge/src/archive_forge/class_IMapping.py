from zope.interface import Interface
from zope.interface.common import collections
class IMapping(IWriteMapping, IEnumerableMapping):
    """ Simple mapping interface """