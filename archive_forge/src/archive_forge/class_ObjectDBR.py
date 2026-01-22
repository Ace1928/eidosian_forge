from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
class ObjectDBR:
    """Defines an interface for object database lookup.
    Objects are identified either by their 20 byte bin sha"""

    def __contains__(self, sha):
        return self.has_obj

    def has_object(self, sha):
        """
        Whether the object identified by the given 20 bytes
            binary sha is contained in the database

        :return: True if the object identified by the given 20 bytes
            binary sha is contained in the database"""
        raise NotImplementedError('To be implemented in subclass')

    def info(self, sha):
        """ :return: OInfo instance
        :param sha: bytes binary sha
        :raise BadObject:"""
        raise NotImplementedError('To be implemented in subclass')

    def stream(self, sha):
        """:return: OStream instance
        :param sha: 20 bytes binary sha
        :raise BadObject:"""
        raise NotImplementedError('To be implemented in subclass')

    def size(self):
        """:return: amount of objects in this database"""
        raise NotImplementedError()

    def sha_iter(self):
        """Return iterator yielding 20 byte shas for all objects in this data base"""
        raise NotImplementedError()