from __future__ import absolute_import
from __future__ import print_function
import threading
import httplib2
from six.moves import range  # pylint: disable=redefined-builtin
def _return_transport(self, transport):
    with self._condition:
        self._transports.append(transport)
        self._condition.notify(n=1)