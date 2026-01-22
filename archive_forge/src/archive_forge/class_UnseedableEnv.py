import warnings
class UnseedableEnv(Error):
    """Raised when the user tries to seed an env that does not support seeding."""