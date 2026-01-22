import copy
from . import ElementTree
from urllib.parse import urljoin
class LimitedRecursiveIncludeError(FatalIncludeError):
    pass