import abc
import datetime
from oslo_utils import timeutils
import keystone.conf
from keystone import exception
class RevokeDriverBase(object, metaclass=abc.ABCMeta):
    """Interface for recording and reporting revocation events."""

    @abc.abstractmethod
    def list_events(self, last_fetch=None, token=None):
        """Return the revocation events, as a list of objects.

        :param last_fetch:   Time of last fetch.  Return all events newer.
        :param token: dictionary of values from a token, normalized for
                      differences between v2 and v3. The checked values are a
                      subset of the attributes of model.TokenEvent
        :returns: A list of keystone.revoke.model.RevokeEvent
                  newer than `last_fetch.`
                  If no last_fetch is specified, returns all events
                  for tokens issued after the expiration cutoff.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def revoke(self, event):
        """Register a revocation event.

        :param event: An instance of
            keystone.revoke.model.RevocationEvent

        """
        raise exception.NotImplemented()