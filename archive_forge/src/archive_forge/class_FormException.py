import calendar
from typing import Any, Optional, Tuple
class FormException(Exception):
    """An error occurred calling the form method."""

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args)
        self.descriptions = kwargs