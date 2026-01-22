from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_function(self, tlist):
    self._last_func = tlist[0]
    self._process_default(tlist)