import copy
from . import ElementTree
from urllib.parse import urljoin
class FatalIncludeError(SyntaxError):
    pass