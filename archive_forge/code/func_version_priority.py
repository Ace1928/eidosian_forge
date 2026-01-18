from pprint import pformat
from six import iteritems
import re
@version_priority.setter
def version_priority(self, version_priority):
    """
        Sets the version_priority of this V1APIServiceSpec.
        VersionPriority controls the ordering of this API version inside of its
        group.  Must be greater than zero. The primary sort is based on
        VersionPriority, ordered highest to lowest (20 before 10). Since it's
        inside of a group, the number can be small, probably in the 10s. In case
        of equal version priorities, the version string will be used to compute
        the order inside a group. If the version string is "kube-like", it
        will sort above non "kube-like" version strings, which are ordered
        lexicographically. "Kube-like" versions start with a "v", then are
        followed by a number (the major version), then optionally the string
        "alpha" or "beta" and another number (the minor version). These are
        sorted first by GA > beta > alpha (where GA is a version with no suffix
        such as beta or alpha), and then by comparing major version, then minor
        version. An example sorted list of versions: v10, v2, v1, v11beta2,
        v10beta3, v3beta1, v12alpha1, v11alpha2, foo1, foo10.

        :param version_priority: The version_priority of this V1APIServiceSpec.
        :type: int
        """
    if version_priority is None:
        raise ValueError('Invalid value for `version_priority`, must not be `None`')
    self._version_priority = version_priority