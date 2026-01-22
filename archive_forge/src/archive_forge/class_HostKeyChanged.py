from twisted.cred.error import UnauthorizedLogin
class HostKeyChanged(Exception):
    """
    The host key of a remote host has changed.

    @ivar offendingEntry: The entry which contains the persistent host key that
    disagrees with the given host key.

    @type offendingEntry: L{twisted.conch.interfaces.IKnownHostEntry}

    @ivar path: a reference to the known_hosts file that the offending entry
    was loaded from

    @type path: L{twisted.python.filepath.FilePath}

    @ivar lineno: The line number of the offending entry in the given path.

    @type lineno: L{int}
    """

    def __init__(self, offendingEntry, path, lineno):
        Exception.__init__(self)
        self.offendingEntry = offendingEntry
        self.path = path
        self.lineno = lineno