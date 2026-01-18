from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Modify the request message, carrying resource name into it.

  Args:
    resource_ref: the resource reference.
    request_msg: the request message constructed by the framework
  Returns:
    the modified request message.
  