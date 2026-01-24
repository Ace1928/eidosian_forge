"""
Utility Functions
---------------
Shared utilities for the Eidos Validator system.
"""

import hashlib
import logging
from datetime import datetime
from importlib import util
from pathlib import Path
from typing import Dict, Any, Optional, List

# Get logger
logger = logging.getLogger("eidos_validator.utils")


def cache_key(prefix: str, *args) -> str:
    """
    Generate cache key from prefix and args.

    Args:
        prefix: String prefix for the key
        *args: Arguments to incorporate into the key

    Returns:
        str: MD5 hash of the key string
    """
    key = prefix + "_".join(str(arg) for arg in args)
    hashed = hashlib.md5(key.encode()).hexdigest()
    logger.debug(f"Generated cache key: {hashed} for prefix: {prefix}")
    return hashed


def check_module_availability(modules: List[str]) -> Dict[str, bool]:
    """
    Check which modules are available in the environment.

    Args:
        modules: List of module names to check

    Returns:
        Dict[str, bool]: Dictionary mapping module names to availability status
    """
    logger.info("Checking module availability...")
    available_modules = {}

    for module_name in modules:
        try:
            if util.find_spec(module_name) is not None:
                __import__(module_name)
                available_modules[module_name] = True
                logger.debug(f"Module {module_name} is available")
            else:
                available_modules[module_name] = False
                logger.warning(f"Module {module_name} not found")
        except ImportError:
            available_modules[module_name] = False
            logger.warning(f"Module {module_name} import failed")
        except Exception as e:
            available_modules[module_name] = False
            logger.error(f"Unexpected error checking {module_name}: {str(e)}")

    logger.info("Module availability check complete")
    return available_modules


def ensure_directories(base_dir: Path, required_dirs: List[str]) -> None:
    """
    Create required directories and verify their accessibility.

    Args:
        base_dir: Base directory path
        required_dirs: List of directory names to create

    Raises:
        IOError: If directory creation fails or directories are not writable
    """
    import os

    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        try:
            if not dir_path.exists():
                logger.info(f"Creating directory: {dir_path}")
                dir_path.mkdir(exist_ok=True, parents=True)
                logger.info(f"Successfully created directory: {dir_path}")
            else:
                # Verify directory permissions and accessibility
                if not os.access(dir_path, os.W_OK):
                    logger.warning(f"Directory exists but not writable: {dir_path}")
                else:
                    logger.debug(f"Verified existing directory: {dir_path}")

            # Create .gitkeep to preserve empty dirs
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()

        except Exception as e:
            logger.error(f"Error creating/verifying directory {dir_path}: {str(e)}")
            raise IOError(f"Failed to create or verify directory {dir_path}: {str(e)}")
