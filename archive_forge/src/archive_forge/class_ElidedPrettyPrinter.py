import textwrap
from pprint import PrettyPrinter
from _plotly_utils.utils import *
from _plotly_utils.data_utils import *
class ElidedPrettyPrinter(PrettyPrinter):
    """
    PrettyPrinter subclass that elides long lists/arrays/strings
    """

    def __init__(self, *args, **kwargs):
        self.threshold = kwargs.pop('threshold', 200)
        PrettyPrinter.__init__(self, *args, **kwargs)

    def _format(self, val, stream, indent, allowance, context, level):
        if ElidedWrapper.is_wrappable(val):
            elided_val = ElidedWrapper(val, self.threshold, indent)
            return self._format(elided_val, stream, indent, allowance, context, level)
        else:
            return PrettyPrinter._format(self, val, stream, indent, allowance, context, level)