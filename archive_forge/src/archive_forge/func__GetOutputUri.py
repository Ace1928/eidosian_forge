from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
def _GetOutputUri(self, poll_result):
    """Gets output uri from poll result.

    Gets output uri from poll result. This is a null implementation that
    returns None. Sub class should override this and return actual output uri
    for output streamer, or returns None if something goes wrong and there is
    no output uri in the poll result.

    Args:
      poll_result: Poll result returned by Poll.

    Returns:
      None. Sub class should override this and returns actual output uri, or
      None when something goes wrong.
    """
    return None