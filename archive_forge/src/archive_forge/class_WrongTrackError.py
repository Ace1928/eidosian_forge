from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import six
class WrongTrackError(DetailedArgumentError):
    """For parsed commands in a different track."""