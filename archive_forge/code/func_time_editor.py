import datetime
from functools import partial
import logging
def time_editor():
    """ Factory function that returns a Time editor for editing Time values.
    """
    from traitsui.api import TimeEditor
    return TimeEditor()