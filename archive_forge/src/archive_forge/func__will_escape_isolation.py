import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
@staticmethod
def _will_escape_isolation(transport_server):
    if not features.paramiko.available():
        return False
    from ....tests import stub_sftp
    if transport_server is stub_sftp.SFTPHomeDirServer:
        return True
    return False