from twisted.cred.error import UnauthorizedLogin
class MissingKeyStoreError(Exception):
    """
    Raised if an SSHAgentServer starts receiving data without its factory
    providing a keys dict on which to read/write key data.
    """