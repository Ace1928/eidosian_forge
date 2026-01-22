import re
import sys
class ClassNotFound(ValueError):
    """Raised if one of the lookup functions didn't find a matching class."""