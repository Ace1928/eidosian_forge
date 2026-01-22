import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _get_sftp_client(self):
    """
        Create SFTP client from the underlying SSH client.

        This method tries to re-use the existing self.sftp_client (if it
        exists) and it also tries to verify the connection is opened and if
        it's not, it will try to re-establish it.
        """
    if not self.sftp_client:
        self.sftp_client = self.client.open_sftp()
    sftp_client = self.sftp_client
    try:
        sftp_client.listdir('.')
    except OSError as e:
        if 'socket is closed' in str(e).lower():
            self.sftp_client = self.client.open_sftp()
        elif 'no such file' in str(e).lower():
            pass
        else:
            raise e
    return self.sftp_client