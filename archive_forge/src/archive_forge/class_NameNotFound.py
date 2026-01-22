import warnings
class NameNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the name doesn't exist."""