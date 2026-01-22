from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import six
class RequiredMutexError(DetailedArgumentError):
    """Required mutex conflict error."""

    def __init__(self, conflict, **kwargs):
        super(RequiredMutexError, self).__init__('Exactly one of {conflict} must be specified.', conflict=conflict, **kwargs)