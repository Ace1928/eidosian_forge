from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLevelIndicator(_messages.Message):
    """A Service-Level Indicator (SLI) describes the "performance" of a
  service. For some services, the SLI is well-defined. In such cases, the SLI
  can be described easily by referencing the well-known SLI and providing the
  needed parameters. Alternatively, a "custom" SLI can be defined with a query
  to the underlying metric store. An SLI is defined to be good_service /
  total_service over any queried time interval. The value of performance
  always falls into the range 0 <= performance <= 1. A custom SLI describes
  how to compute this ratio, whether this is by dividing values from a pair of
  time series, cutting a Distribution into good and bad counts, or counting
  time windows in which the service complies with a criterion. For separation
  of concerns, a single Service-Level Indicator measures performance for only
  one aspect of service quality, such as fraction of successful queries or
  fast-enough queries.

  Fields:
    basicSli: Basic SLI on a well-known service type.
    requestBased: Request-based SLIs
    windowsBased: Windows-based SLIs
  """
    basicSli = _messages.MessageField('BasicSli', 1)
    requestBased = _messages.MessageField('RequestBasedSli', 2)
    windowsBased = _messages.MessageField('WindowsBasedSli', 3)