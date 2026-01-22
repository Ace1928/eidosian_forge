import warnings
class UnregisteredBenchmark(Unregistered):
    """Raised when the user requests an env from the registry that does not actually exist."""