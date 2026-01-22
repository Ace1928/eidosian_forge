from ._base import *
from .models import LazyDBCacheBase
@property
def all_wks(self):
    return list(self.sheet.worksheets())