import os
import random
import warnings
class SourceNotAvailable(RuntimeError):
    """
    Internal exception used when a specific random source is not available.
    """