import copy
import errno
import os
import sys
import ovs.dirs
import ovs.jsonrpc
import ovs.stream
import ovs.unixctl
import ovs.util
import ovs.version
import ovs.vlog
def _unixctl_version(conn, unused_argv, version):
    assert isinstance(conn, UnixctlConnection)
    version = '%s (Open vSwitch) %s' % (ovs.util.PROGRAM_NAME, version)
    conn.reply(version)