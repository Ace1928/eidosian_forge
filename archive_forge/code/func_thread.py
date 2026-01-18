import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def thread(self, threading_algorithm, charset, *search_criteria):
    """IMAPrev1 extension THREAD command.

        (type, [data]) = <instance>.thread(threading_algorithm, charset, search_criteria, ...)
        """
    name = 'THREAD'
    typ, dat = self._simple_command(name, threading_algorithm, charset, *search_criteria)
    return self._untagged_response(typ, dat, name)