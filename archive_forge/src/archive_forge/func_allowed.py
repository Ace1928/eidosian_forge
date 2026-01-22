from pprint import pformat
from six import iteritems
import re
@allowed.setter
def allowed(self, allowed):
    """
        Sets the allowed of this V1SubjectAccessReviewStatus.
        Allowed is required. True if the action would be allowed, false
        otherwise.

        :param allowed: The allowed of this V1SubjectAccessReviewStatus.
        :type: bool
        """
    if allowed is None:
        raise ValueError('Invalid value for `allowed`, must not be `None`')
    self._allowed = allowed