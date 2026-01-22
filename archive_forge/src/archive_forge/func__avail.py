import csv
import datetime
import os
def _avail(self, date):
    """Return all distributions that were available on the given date."""
    return [x for x in self._releases if date >= x.created]