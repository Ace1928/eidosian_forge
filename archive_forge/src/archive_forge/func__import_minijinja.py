from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union
from ..utils import HfHubHTTPError, RepositoryNotFoundError, is_minijinja_available
def _import_minijinja():
    if not is_minijinja_available():
        raise ImportError('Cannot render template. Please install minijinja using `pip install minijinja`.')
    import minijinja
    return minijinja