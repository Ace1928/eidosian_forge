from ._base import *
from .models import LazyDBCacheBase
@property
def all_wks_dict(self):
    return {n: s for n, s in enumerate(self.all_wks)}