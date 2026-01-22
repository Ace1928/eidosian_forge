from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import logging
import multiprocessing
import threading
import traceback
from gslib.utils import constants
from gslib.utils import system_util
from six.moves import queue as Queue
def ShouldProhibitMultiprocessing():
    """Determines if the OS doesn't support multiprocessing.

  There are two cases we currently know about:
    - Multiple processes are not supported on Windows.
    - If an error is encountered while using multiple processes on Alpine Linux
      gsutil hangs. For this case it's possible we could do more work to find
      the root cause but after a fruitless initial attempt we decided instead
      to fall back on multi-threading w/o multiprocesing.

  Returns:
    (bool indicator if multiprocessing should be prohibited, OS name)
  """
    if system_util.IS_WINDOWS:
        return (True, 'Windows')
    if system_util.IS_OSX:
        return (False, 'macOS')
    try:
        with open('/etc/os-release', 'r') as f:
            for line in f.read().splitlines():
                if 'NAME=' in line:
                    os_name = line.split('=')[1].strip('"')
                    return ('alpine linux' in os_name.lower(), os_name)
            return (False, 'Unknown')
    except IOError as e:
        if e.errno == errno.ENOENT:
            logging.debug('Unable to open /etc/os-release to determine whether OS supports multiprocessing: errno=%d, message=%s' % (e.errno, str(e)))
            return (False, 'Unknown')
        else:
            raise
    except Exception as exc:
        logging.debug('Something went wrong while trying to determine multiprocessing capabilities.\nMessage: {0}'.format(str(exc)))
        return (False, 'Unknown')