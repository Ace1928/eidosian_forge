from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
Implements apitools.base.protorpclite.util.TimeZoneOffset.__getinitargs__.

  The apitools TimeZoneOffset class inherits from datetime.datetime, which
  implements custom picking behavior in __reduce__. This reduce method cannot
  handle the additional argument that TimeZoneOffset adds to __init__, which
  makes TimeZoneOffset unpicklable without implementing __getinitargs__ as
  we do here.

  Args:
    self (TimeZoneOffset): an instance of TimeZoneOffset.

  Returns:
    A tuple of arguments passed to TimeZoneOffset.__init__ when unpickling.
  