from pprint import pformat
from six import iteritems
import re
@ttl_seconds_after_finished.setter
def ttl_seconds_after_finished(self, ttl_seconds_after_finished):
    """
        Sets the ttl_seconds_after_finished of this V1JobSpec.
        ttlSecondsAfterFinished limits the lifetime of a Job that has finished
        execution (either Complete or Failed). If this field is set,
        ttlSecondsAfterFinished after the Job finishes, it is eligible to be
        automatically deleted. When the Job is being deleted, its lifecycle
        guarantees (e.g. finalizers) will be honored. If this field is unset,
        the Job won't be automatically deleted. If this field is set to zero,
        the Job becomes eligible to be deleted immediately after it finishes.
        This field is alpha-level and is only honored by servers that enable the
        TTLAfterFinished feature.

        :param ttl_seconds_after_finished: The ttl_seconds_after_finished of
        this V1JobSpec.
        :type: int
        """
    self._ttl_seconds_after_finished = ttl_seconds_after_finished