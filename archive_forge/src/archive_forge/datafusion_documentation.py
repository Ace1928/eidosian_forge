from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
Stateful utility for calling Datafusion APIs.

  While this currently could all be stati, it is encapsulated in a class to
  support API version switching in future.
  