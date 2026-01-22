import importlib
import math
import re
from enum import Enum
class ReportResult(Enum):
    """
    Result of filing a report.

    FAILURE:    a player timed out while reporting, or it was an accidental report
    BLOCK:      a player is blocked, for having been reported > 1 times
    SUCCESS:    a successful report
    BOT:        the offending agent was the bot
    """
    FAILURE = 0
    BLOCK = 1
    SUCCESS = 2
    BOT = 3