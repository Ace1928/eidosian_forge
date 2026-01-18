import os
from paramiko.sftp import SFTP_OP_UNSUPPORTED, SFTP_OK
from paramiko.util import ClosingContextManager
from paramiko.sftp_server import SFTPServer

        Used by the SFTP server code to retrieve a cached directory
        listing.
        