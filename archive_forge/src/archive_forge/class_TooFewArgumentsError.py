from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import six
class TooFewArgumentsError(ArgumentError):
    """Argparse didn't use all the Positional objects."""