import threading
from oslo_utils import timeutils
Waits until the latch is released.

        :param timeout: wait until the timeout expires
        :type timeout: number
        :returns: true if the latch has been released before the
                  timeout expires otherwise false
        :rtype: boolean
        