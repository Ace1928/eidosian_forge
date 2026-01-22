import socket
class BadHostKeyException(SSHException):
    """
    The host key given by the SSH server did not match what we were expecting.

    :param str hostname: the hostname of the SSH server
    :param PKey got_key: the host key presented by the server
    :param PKey expected_key: the host key expected

    .. versionadded:: 1.6
    """

    def __init__(self, hostname, got_key, expected_key):
        SSHException.__init__(self, hostname, got_key, expected_key)
        self.hostname = hostname
        self.key = got_key
        self.expected_key = expected_key

    def __str__(self):
        msg = "Host key for server '{}' does not match: got '{}', expected '{}'"
        return msg.format(self.hostname, self.key.get_base64(), self.expected_key.get_base64())