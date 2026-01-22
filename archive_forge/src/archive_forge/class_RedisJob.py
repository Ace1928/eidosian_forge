import contextlib
import datetime
import functools
import re
import string
import threading
import time
import fasteners
import msgpack
from oslo_serialization import msgpackutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from redis import exceptions as redis_exceptions
from redis import sentinel
from taskflow import exceptions as exc
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import misc
from taskflow.utils import redis_utils as ru
@functools.total_ordering
class RedisJob(base.Job):
    """A redis job."""

    def __init__(self, board, name, sequence, key, uuid=None, details=None, created_on=None, backend=None, book=None, book_data=None, priority=base.JobPriority.NORMAL):
        super(RedisJob, self).__init__(board, name, uuid=uuid, details=details, backend=backend, book=book, book_data=book_data)
        self._created_on = created_on
        self._client = board._client
        self._redis_version = board._redis_version
        self._sequence = sequence
        self._key = key
        self._last_modified_key = board.join(key + board.LAST_MODIFIED_POSTFIX)
        self._owner_key = board.join(key + board.OWNED_POSTFIX)
        self._priority = priority

    @property
    def key(self):
        """Key (in board listings/trash hash) the job data is stored under."""
        return self._key

    @property
    def priority(self):
        return self._priority

    @property
    def last_modified_key(self):
        """Key the job last modified data is stored under."""
        return self._last_modified_key

    @property
    def owner_key(self):
        """Key the job claim + data of the owner is stored under."""
        return self._owner_key

    @property
    def sequence(self):
        """Sequence number of the current job."""
        return self._sequence

    def expires_in(self):
        """How many seconds until the claim expires.

        Returns the number of seconds until the ownership entry expires or
        :attr:`~taskflow.utils.redis_utils.UnknownExpire.DOES_NOT_EXPIRE` or
        :attr:`~taskflow.utils.redis_utils.UnknownExpire.KEY_NOT_FOUND` if it
        does not expire or if the expiry can not be determined (perhaps the
        :attr:`.owner_key` expired at/before time of inquiry?).
        """
        with _translate_failures():
            return ru.get_expiry(self._client, self._owner_key, prior_version=self._redis_version)

    def extend_expiry(self, expiry):
        """Extends the owner key (aka the claim) expiry for this job.

        NOTE(harlowja): if the claim for this job did **not** previously
        have an expiry associated with it, calling this method will create
        one (and after that time elapses the claim on this job will cease
        to exist).

        Returns ``True`` if the expiry request was performed
        otherwise ``False``.
        """
        with _translate_failures():
            return ru.apply_expiry(self._client, self._owner_key, expiry, prior_version=self._redis_version)

    def __lt__(self, other):
        if not isinstance(other, RedisJob):
            return NotImplemented
        if self.board.listings_key == other.board.listings_key:
            if self.priority == other.priority:
                return self.sequence < other.sequence
            else:
                ordered = base.JobPriority.reorder((self.priority, self), (other.priority, other))
                if ordered[0] is self:
                    return False
                return True
        else:
            return self.board.listings_key < other.board.listings_key

    def __eq__(self, other):
        if not isinstance(other, RedisJob):
            return NotImplemented
        return (self.board.listings_key, self.priority, self.sequence) == (other.board.listings_key, other.priority, other.sequence)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.board.listings_key, self.priority, self.sequence))

    @property
    def created_on(self):
        return self._created_on

    @property
    def last_modified(self):
        with _translate_failures():
            raw_last_modified = self._client.get(self._last_modified_key)
        last_modified = None
        if raw_last_modified:
            last_modified = self._board._loads(raw_last_modified, root_types=(datetime.datetime,))
            last_modified = max(last_modified, self._created_on)
        return last_modified

    @property
    def state(self):
        listings_key = self._board.listings_key
        owner_key = self._owner_key
        listings_sub_key = self._key

        def _do_fetch(p):
            p.multi()
            p.hexists(listings_key, listings_sub_key)
            p.exists(owner_key)
            job_exists, owner_exists = p.execute()
            if not job_exists:
                if owner_exists:
                    LOG.info("Unexpected owner key found at '%s' when job key '%s[%s]' was not found", owner_key, listings_key, listings_sub_key)
                return states.COMPLETE
            elif owner_exists:
                return states.CLAIMED
            else:
                return states.UNCLAIMED
        with _translate_failures():
            return self._client.transaction(_do_fetch, listings_key, owner_key, value_from_callable=True)