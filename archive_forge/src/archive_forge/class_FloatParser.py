from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class FloatParser(NumericParser):
    """Parser of floating point values.

  Parsed value may be bounded to a given upper and lower bound.
  """
    number_article = 'a'
    number_name = 'number'
    syntactic_help = ' '.join((number_article, number_name))

    def __init__(self, lower_bound=None, upper_bound=None):
        super(FloatParser, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        sh = self.syntactic_help
        if lower_bound is not None and upper_bound is not None:
            sh = '%s in the range [%s, %s]' % (sh, lower_bound, upper_bound)
        elif lower_bound == 0:
            sh = 'a non-negative %s' % self.number_name
        elif upper_bound == 0:
            sh = 'a non-positive %s' % self.number_name
        elif upper_bound is not None:
            sh = '%s <= %s' % (self.number_name, upper_bound)
        elif lower_bound is not None:
            sh = '%s >= %s' % (self.number_name, lower_bound)
        self.syntactic_help = sh

    def convert(self, argument):
        """Returns the float value of argument."""
        if _is_integer_type(argument) or isinstance(argument, float) or isinstance(argument, six.string_types):
            return float(argument)
        else:
            raise TypeError('Expect argument to be a string, int, or float, found {}'.format(type(argument)))

    def flag_type(self):
        """See base class."""
        return 'float'