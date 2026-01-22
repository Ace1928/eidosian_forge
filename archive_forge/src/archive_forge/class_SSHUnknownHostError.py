from ncclient import NCClientError
class SSHUnknownHostError(SSHError):

    def __init__(self, host, fingerprint):
        SSHError.__init__(self, 'Unknown host key [%s] for [%s]' % (fingerprint, host))
        self.host = host
        self.fingerprint = fingerprint