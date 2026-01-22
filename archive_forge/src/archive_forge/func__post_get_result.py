import re
from ._exceptions import ProgrammingError
def _post_get_result(self):
    self._rows = self._fetch_row(0)
    self._result = None