from typing import Optional
class QiskitUserConfigError(QiskitError):
    """Raised when an error is encountered reading a user config file."""
    message = 'User config invalid'