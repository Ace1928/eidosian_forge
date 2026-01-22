from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import six
class OptionalMutexError(DetailedArgumentError):
    """Optional mutex conflict error."""

    def __init__(self, conflict, **kwargs):
        super(OptionalMutexError, self).__init__('At most one of {conflict} can be specified.', conflict=conflict, **kwargs)