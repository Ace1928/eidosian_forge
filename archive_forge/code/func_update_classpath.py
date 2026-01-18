import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, overload
from scrapy.exceptions import ScrapyDeprecationWarning
def update_classpath(path: Any) -> Any:
    """Update a deprecated path from an object with its new location"""
    for prefix, replacement in DEPRECATION_RULES:
        if isinstance(path, str) and path.startswith(prefix):
            new_path = path.replace(prefix, replacement, 1)
            warnings.warn(f'`{path}` class is deprecated, use `{new_path}` instead', ScrapyDeprecationWarning)
            return new_path
    return path