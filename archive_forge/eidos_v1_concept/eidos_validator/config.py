"""
Configuration for Eidos Validator
--------------------------------
Contains constants, defaults, and configuration values for the Eidos Validator system.
"""

import os
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, List

# Base directory is the parent of the eidos_validator package
BASE_DIR = Path(__file__).parent.parent

# Cache configuration
CACHE_DIR = BASE_DIR / ".cache"

# Cache expiry times
CACHE_EXPIRY = {
    "static": timedelta(days=7),  # Hardware/OS info that rarely changes
    "dynamic": timedelta(hours=1),  # Memory/CPU/network stats that change frequently
    "volatile": timedelta(minutes=5),  # Very dynamic data like CPU usage
}

# Required directories from schema
REQUIRED_DIRS = [
    "cycles",
    "metadata",
    "synthesis",
    "relationship",
    "memory",
    "consciousness",
    "evolution",
    "emotional",
    "cognitive",
    "introspection",
    "self_model",
    "qualia",
    "learning",
    "adaptation",
    "archive",
    "backup",
    "temp",  # Additional utility directories
]

# Default storage configuration
DEFAULT_STORAGE_CONFIG = {
    "base_path": str(CACHE_DIR),
    "file_structure": {
        "cycles_dir": "cycles",
        "metadata_dir": "metadata",
        "synthesis_dir": "synthesis",
        "relationship_dir": "relationship",
        "memory_dir": "memory",
        "consciousness_dir": "consciousness",
        "evolution_dir": "evolution",
        "emotional_dir": "emotional",
        "cognitive_dir": "cognitive",
        "introspection_dir": "introspection",
        "self_model_dir": "self_model",
        "qualia_dir": "qualia",
        "learning_dir": "learning",
        "adaptation_dir": "adaptation",
        "archive_dir": "archive",
        "backup_dir": "backup",
        "temp_dir": "temp",
    },
    "retention_policy": {
        "min_retention_period": "30 days",
        "archival_strategy": "compress_and_archive",
        "backup_frequency": "1 day",
        "data_importance_levels": {
            "critical": "never_delete",
            "important": "5 years",
            "routine": "6 months",
            "developmental": "2 years",
            "experiential": "3 months",
            "emotional": "1 year",
            "cognitive": "1 year",
            "consciousness": "permanent",
            "identity": "permanent",
        },
    },
    "file_handling": {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "compression": True,
        "compression_level": 6,
        "backup_enabled": True,
        "encryption_enabled": False,
    },
}

# Dictionary of available modules to check
REQUIRED_MODULES = [
    # Core functionality
    "json",
    "jsonschema",
    "logging",
    "pathlib",
    "typing",
    # System info gathering
    "platform",
    "psutil",
    "os",
    "distro",
    "subprocess",
    # Network functionality
    "socket",
    "requests",
    "ssl",
    # Utilities
    "datetime",
    "uuid",
    "shutil",
    "zlib",
    # Optional ML/AI libraries
    "numpy",
    "tensorflow",
    "torch",
    "pandas",
    "sklearn",
]
