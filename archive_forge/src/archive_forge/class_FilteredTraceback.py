import functools
import traceback
class FilteredTraceback(object):
    """Wraps a traceback to filter unwanted frames."""

    def __init__(self, tb, filtered_traceback):
        """Constructor.

        :param tb: The start of the traceback chain to filter.
        :param filtered_traceback: The first traceback of a trailing
               chain that is to be filtered.
        """
        self._tb = tb
        self.tb_lasti = self._tb.tb_lasti
        self.tb_lineno = self._tb.tb_lineno
        self.tb_frame = self._tb.tb_frame
        self._filtered_traceback = filtered_traceback

    @property
    def tb_next(self):
        tb_next = self._tb.tb_next
        if tb_next and tb_next != self._filtered_traceback:
            return FilteredTraceback(tb_next, self._filtered_traceback)