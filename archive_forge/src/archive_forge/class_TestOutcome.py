from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
class TestOutcome(collections.namedtuple('TestOutcome', ['outcome', 'axis_value', 'test_details'])):
    """A tuple to hold the outcome for a single test axis value.

  Fields:
    outcome: string containing the test outcome (e.g. 'Passed')
    axis_value: string representing one axis value within the matrix.
    test_details: string with extra details (e.g. "Incompatible architecture")
  """