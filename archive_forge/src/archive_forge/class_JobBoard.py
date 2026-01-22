import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
class JobBoard(object, metaclass=abc.ABCMeta):
    """A place where jobs can be posted, reposted, claimed and transferred.

    There can be multiple implementations of this job board, depending on the
    desired semantics and capabilities of the underlying jobboard
    implementation.

    NOTE(harlowja): the name is meant to be an analogous to a board/posting
    system that is used in newspapers, or elsewhere to solicit jobs that
    people can interview and apply for (and then work on & complete).
    """

    def __init__(self, name, conf):
        self._name = name
        self._conf = conf

    @abc.abstractmethod
    def iterjobs(self, only_unclaimed=False, ensure_fresh=False):
        """Returns an iterator of jobs that are currently on this board.

        NOTE(harlowja): the ordering of this iteration should be by posting
        order (oldest to newest) with higher priority jobs
        being provided before lower priority jobs, but it is left up to the
        backing implementation to provide the order that best suits it..

        NOTE(harlowja): the iterator that is returned may support other
        attributes which can be used to further customize how iteration can
        be accomplished; check with the backends iterator object to determine
        what other attributes are supported.

        :param only_unclaimed: boolean that indicates whether to only iteration
            over unclaimed jobs.
        :param ensure_fresh: boolean that requests to only iterate over the
            most recent jobs available, where the definition of what is recent
            is backend specific. It is allowable that a backend may ignore this
            value if the backends internal semantics/capabilities can not
            support this argument.
        """

    @abc.abstractmethod
    def wait(self, timeout=None):
        """Waits a given amount of time for **any** jobs to be posted.

        When jobs are found then an iterator will be returned that can be used
        to iterate over those jobs.

        NOTE(harlowja): since a jobboard can be mutated on by multiple external
        entities at the **same** time the iterator that can be
        returned **may** still be empty due to other entities removing those
        jobs after the iterator has been created (be aware of this when
        using it).

        :param timeout: float that indicates how long to wait for a job to
            appear (if None then waits forever).
        """

    @property
    @abc.abstractmethod
    def job_count(self):
        """Returns how many jobs are on this jobboard.

        NOTE(harlowja): this count may change as jobs appear or are removed so
        the accuracy of this count should not be used in a way that requires
        it to be exact & absolute.
        """

    @abc.abstractmethod
    def find_owner(self, job):
        """Gets the owner of the job if one exists."""

    @property
    def name(self):
        """The non-uniquely identifying name of this jobboard."""
        return self._name

    @abc.abstractmethod
    def consume(self, job, who):
        """Permanently (and atomically) removes a job from the jobboard.

        Consumption signals to the board (and any others examining the board)
        that this job has been completed by the entity that previously claimed
        that job.

        Only the entity that has claimed that job is able to consume the job.

        A job that has been consumed can not be reclaimed or reposted by
        another entity (job postings are immutable). Any entity consuming
        a unclaimed job (or a job they do not have a claim on) will cause an
        exception.

        :param job: a job on this jobboard that can be consumed (if it does
            not exist then a NotFound exception will be raised).
        :param who: string that names the entity performing the consumption,
            this must be the same name that was used for claiming this job.
        """

    @abc.abstractmethod
    def post(self, name, book=None, details=None, priority=JobPriority.NORMAL):
        """Atomically creates and posts a job to the jobboard.

        This posting allowing others to attempt to claim that job (and
        subsequently work on that job). The contents of the provided logbook,
        details dictionary, or name (or a mix of these) must provide *enough*
        information for consumers to reference to construct and perform that
        jobs contained work (whatever it may be).

        Once a job has been posted it can only be removed by consuming that
        job (after that job is claimed). Any entity can post/propose jobs
        to the jobboard (in the future this may be restricted).

        Returns a job object representing the information that was posted.
        """

    @abc.abstractmethod
    def claim(self, job, who):
        """Atomically attempts to claim the provided job.

        If a job is claimed it is expected that the entity that claims that job
        will at sometime in the future work on that jobs contents and either
        fail at completing them (resulting in a reposting) or consume that job
        from the jobboard (signaling its completion). If claiming fails then
        a corresponding exception will be raised to signal this to the claim
        attempter.

        :param job: a job on this jobboard that can be claimed (if it does
            not exist then a NotFound exception will be raised).
        :param who: string that names the claiming entity.
        """

    @abc.abstractmethod
    def abandon(self, job, who):
        """Atomically attempts to abandon the provided job.

        This abandonment signals to others that the job may now be reclaimed.
        This would typically occur if the entity that has claimed the job has
        failed or is unable to complete the job or jobs it had previously
        claimed.

        Only the entity that has claimed that job can abandon a job. Any entity
        abandoning a unclaimed job (or a job they do not own) will cause an
        exception.

        :param job: a job on this jobboard that can be abandoned (if it does
            not exist then a NotFound exception will be raised).
        :param who: string that names the entity performing the abandoning,
            this must be the same name that was used for claiming this job.
        """

    @abc.abstractmethod
    def trash(self, job, who):
        """Trash the provided job.

        Trashing a job signals to others that the job is broken and should not
        be reclaimed. This is provided as an option for users to be able to
        remove jobs from the board externally.  The trashed job details should
        be kept around in an alternate location to be reviewed, if desired.

        Only the entity that has claimed that job can trash a job. Any entity
        trashing a unclaimed job (or a job they do not own) will cause an
        exception.

        :param job: a job on this jobboard that can be trashed (if it does
            not exist then a NotFound exception will be raised).
        :param who: string that names the entity performing the trashing,
            this must be the same name that was used for claiming this job.
        """

    @abc.abstractmethod
    def register_entity(self, entity):
        """Register an entity to the jobboard('s backend), e.g: a conductor.

        :param entity: entity to register as being associated with the
                       jobboard('s backend)
        :type entity: :py:class:`~taskflow.types.entity.Entity`
        """

    @property
    @abc.abstractmethod
    def connected(self):
        """Returns if this jobboard is connected."""

    @abc.abstractmethod
    def connect(self):
        """Opens the connection to any backend system."""

    @abc.abstractmethod
    def close(self):
        """Close the connection to any backend system.

        Once closed the jobboard can no longer be used (unless reconnection
        occurs).
        """