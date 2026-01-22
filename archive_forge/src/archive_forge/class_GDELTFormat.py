from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
class GDELTFormat(Enum):
    dict = 'dict'
    obj = 'obj'
    json = 'json'
    pandas = 'pd'