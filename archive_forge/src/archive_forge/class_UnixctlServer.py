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
class UnixctlServer(object):

    def __init__(self, listener):
        assert isinstance(listener, ovs.stream.PassiveStream)
        self._listener = listener
        self._conns = []

    def run(self):
        for _ in range(10):
            error, stream = self._listener.accept()
            if sys.platform == 'win32' and error == errno.WSAEWOULDBLOCK:
                error = errno.EAGAIN
            if not error:
                rpc = ovs.jsonrpc.Connection(stream)
                self._conns.append(UnixctlConnection(rpc))
            elif error == errno.EAGAIN:
                break
            else:
                vlog.warn('%s: accept failed: %s' % (self._listener.name, os.strerror(error)))
        for conn in copy.copy(self._conns):
            error = conn.run()
            if error and error != errno.EAGAIN:
                conn._close()
                self._conns.remove(conn)

    def wait(self, poller):
        self._listener.wait(poller)
        for conn in self._conns:
            conn._wait(poller)

    def close(self):
        for conn in self._conns:
            conn._close()
        self._conns = None
        self._listener.close()
        self._listener = None

    @staticmethod
    def create(path, version=None):
        """Creates a new UnixctlServer which listens on a unixctl socket
        created at 'path'.  If 'path' is None, the default path is chosen.
        'version' contains the version of the server as reported by the unixctl
        version command.  If None, ovs.version.VERSION is used."""
        assert path is None or isinstance(path, str)
        if path is not None:
            path = 'punix:%s' % ovs.util.abs_file_name(ovs.dirs.RUNDIR, path)
        elif sys.platform == 'win32':
            path = 'punix:%s/%s.ctl' % (ovs.dirs.RUNDIR, ovs.util.PROGRAM_NAME)
        else:
            path = 'punix:%s/%s.%d.ctl' % (ovs.dirs.RUNDIR, ovs.util.PROGRAM_NAME, os.getpid())
        if version is None:
            version = ovs.version.VERSION
        error, listener = ovs.stream.PassiveStream.open(path)
        if error:
            ovs.util.ovs_error(error, 'could not initialize control socket %s' % path)
            return (error, None)
        ovs.unixctl.command_register('version', '', 0, 0, _unixctl_version, version)
        return (0, UnixctlServer(listener))