import logging
import os
class InvalidBuiltinName(Exception):
    """Raised whenever a builtin handler name is specified that is not found."""