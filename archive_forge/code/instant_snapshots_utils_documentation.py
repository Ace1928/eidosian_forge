from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
Gets the zonal or regional instant snapshot api info.

  Args:
    ips_ref: the instant snapshot resource reference that is parsed from
      resource arguments.
    client: the compute api_tools_client.
    messages: the compute message module.

  Returns:
    _ZoneInstantSnapshot or _RegionInstantSnapshot.
  