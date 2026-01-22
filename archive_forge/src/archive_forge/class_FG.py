import os
import sys
class FG:
    """Unix terminal foreground color codes (16-color)."""
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    MAGENTA = '\x1b[35m'
    CYAN = '\x1b[36m'
    WHITE = '\x1b[37m'
    BOLD_RED = '\x1b[1;31m'
    BOLD_GREEN = '\x1b[1;32m'
    BOLD_YELLOW = '\x1b[1;33m'
    BOLD_BLUE = '\x1b[1;34m'
    BOLD_MAGENTA = '\x1b[1;35m'
    BOLD_CYAN = '\x1b[1;36m'
    BOLD_WHITE = '\x1b[1;37m'
    NONE = '\x1b[0m'