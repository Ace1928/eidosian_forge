import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def unselect(self):
    """Free server's resources associated with the selected mailbox
        and returns the server to the authenticated state.
        This command performs the same actions as CLOSE, except
        that no messages are permanently removed from the currently
        selected mailbox.

        (typ, [data]) = <instance>.unselect()
        """
    try:
        typ, data = self._simple_command('UNSELECT')
    finally:
        self.state = 'AUTH'
    return (typ, data)