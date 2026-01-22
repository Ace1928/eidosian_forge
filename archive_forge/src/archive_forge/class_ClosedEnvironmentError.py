import warnings
class ClosedEnvironmentError(Exception):
    """Trying to call `reset`, or `step`, while the environment is closed."""