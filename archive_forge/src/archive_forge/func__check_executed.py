import re
import warnings
from . import err
def _check_executed(self):
    if not self._executed:
        raise err.ProgrammingError('execute() first')