import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def _try_pkey_auth(paramiko_transport, pkey_class, username, filename):
    filename = os.path.expanduser('~/.ssh/' + filename)
    try:
        key = pkey_class.from_private_key_file(filename)
        paramiko_transport.auth_publickey(username, key)
        return True
    except paramiko.PasswordRequiredException:
        password = ui.ui_factory.get_password(prompt='SSH %(filename)s password', filename=os.fsdecode(filename))
        try:
            key = pkey_class.from_private_key_file(filename, password)
            paramiko_transport.auth_publickey(username, key)
            return True
        except paramiko.SSHException:
            trace.mutter('SSH authentication via %s key failed.' % (os.path.basename(filename),))
    except paramiko.SSHException:
        trace.mutter('SSH authentication via %s key failed.' % (os.path.basename(filename),))
    except OSError:
        pass
    return False