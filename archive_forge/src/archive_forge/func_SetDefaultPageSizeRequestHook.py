from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SetDefaultPageSizeRequestHook(default_page_size):
    """Create a modify_request_hook that applies default_page_size to args.

  Args:
    default_page_size: The page size to use when not specified by the user.

  Returns:
    A modify_request_hook that updates `args.page_size` when not set by user.
  """

    def Hook(unused_ref, args, request):
        if not args.page_size:
            args.page_size = int(default_page_size)
        return request
    return Hook