import warnings
class DeprecatedEnv(Error):
    """Raised when the user requests an env from the registry with an older version number than the latest env with the same name."""