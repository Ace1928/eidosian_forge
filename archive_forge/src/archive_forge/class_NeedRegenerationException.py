import logging
import time
class NeedRegenerationException(Exception):
    """An exception that when raised in the 'with' block,
    forces the 'has_value' flag to False and incurs a
    regeneration of the value.

    """