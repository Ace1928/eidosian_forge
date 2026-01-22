import sys
class NoKeyringError(KeyringError, RuntimeError):
    """Raised when there is no keyring backend"""