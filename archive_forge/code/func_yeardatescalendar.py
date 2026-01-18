import sys
import datetime
import locale as _locale
from itertools import repeat
def yeardatescalendar(self, year, width=3):
    """
        Return the data for the specified year ready for formatting. The return
        value is a list of month rows. Each month row contains up to width months.
        Each month contains between 4 and 6 weeks and each week contains 1-7
        days. Days are datetime.date objects.
        """
    months = [self.monthdatescalendar(year, i) for i in range(January, January + 12)]
    return [months[i:i + width] for i in range(0, len(months), width)]