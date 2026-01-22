import warnings
class ResetNotAllowed(Error):
    """When the monitor is active, raised when the user tries to step an environment that's not yet terminated or truncated."""