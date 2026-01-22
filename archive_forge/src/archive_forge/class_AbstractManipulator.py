import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
class AbstractManipulator(object, metaclass=abc.ABCMeta):
    """Abstract class with methods which need to be implemented for custom
    manipulation.

    Adding this as a base class for :class:`.BaseTransform` instead of adding
    import dependency of pymongo specific class i.e.
    `pymongo.son_manipulator.SONManipulator` and using that as base class.
    This is done to avoid pymongo dependency if MongoDB backend is not used.
    """

    @abc.abstractmethod
    def transform_incoming(self, son, collection):
        """Used while saving data to MongoDB.

        :param son: the SON object to be inserted into the database
        :param collection: the collection the object is being inserted into

        :returns: transformed SON object

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_outgoing(self, son, collection):
        """Used while reading data from MongoDB.

        :param son: the SON object being retrieved from the database
        :param collection: the collection this object was stored in

        :returns: transformed SON object
        """
        raise NotImplementedError()

    def will_copy(self):
        """Will this SON manipulator make a copy of the incoming document?

        Derived classes that do need to make a copy should override this
        method, returning `True` instead of `False`.

        :returns: boolean
        """
        return False