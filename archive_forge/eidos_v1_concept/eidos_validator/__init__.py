"""
Eidos Validator Package
-----------------------
A comprehensive system for validating, enhancing, and managing Eidosian data structures.

This package provides tools for JSON validation, system information gathering,
caching, and data storage according to the Eidosian schema structure.
"""

__version__ = "1.0.0"

# Core components
from .core import EidosValidator

# Specialized validators and enhancers
from .validator import SchemaValidator
from .enhancer import JsonEnhancer

# System and infrastructure
from .cache import CacheManager, ThreadSafeCache
from .storage import StorageManager, RolloverFileManager
from .system import SystemInfoCollector
from .network import NetworkInfoCollector

# Configuration and utilities
from .config import BASE_DIR, CACHE_DIR, REQUIRED_DIRS
from .logging_config import configure_logging
from .utils import cache_key, check_module_availability, ensure_directories
