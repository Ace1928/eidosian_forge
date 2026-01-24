"""
Cache Management
--------------
Provides caching mechanisms for Eidos Validator.
"""

import logging
import pickle
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .config import CACHE_DIR, CACHE_EXPIRY

# Get logger
logger = logging.getLogger("eidos_validator.cache")


class ThreadSafeCache:
    """
    Thread-safe in-memory cache with fallback to non-threaded mode.
    """

    def __init__(self) -> None:
        """Initialize the thread-safe cache."""
        self._cache: Dict[str, Any] = {}
        try:
            self._lock = threading.Lock()
            self._threaded = True
            logger.info("Initialized thread-safe cache with locking enabled")
        except (ImportError, RuntimeError):
            logger.warning("Threading not available, using non-threaded mode")
            self._threaded = False

    def get(self, key: str) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Any: The cached value or None if not found
        """
        if self._threaded:
            with self._lock:
                return self._cache.get(key)
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if self._threaded:
            with self._lock:
                self._cache[key] = value
        else:
            self._cache[key] = value

    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.

        Args:
            key: Cache key
        """
        if self._threaded:
            with self._lock:
                self._cache.pop(key, None)
        else:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all items from the cache."""
        if self._threaded:
            with self._lock:
                self._cache.clear()
        else:
            self._cache.clear()


class CacheManager:
    """
    Manages disk-based caching with expiry.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to use for cache files (default: config.CACHE_DIR)
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self._ensure_cache_directory()
        self.memory_cache = ThreadSafeCache()

    def _ensure_cache_directory(self) -> None:
        """Ensure the cache directory exists and is writable."""
        logger.info(f"Initializing cache directory at: {self.cache_dir}")
        try:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            # Also create temp dir for atomic writes
            (self.cache_dir / "temp").mkdir(exist_ok=True)
            logger.info(f"Cache directory created/verified at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {str(e)}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """
        Get data from disk cache with expiry check.

        Args:
            key: The cache key

        Returns:
            Optional[Any]: The cached value or None if not found/expired
        """
        # First check memory cache
        memory_result = self.memory_cache.get(key)
        if memory_result is not None:
            logger.debug(f"Memory cache hit: {key}")
            return memory_result

        cache_file = self.cache_dir / f"{key}.cache"

        if not cache_file.exists():
            logger.debug(f"Cache miss: {key}")
            return None

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            if datetime.now() > data["expiry"]:
                logger.debug(f"Cache expired: {key}")
                try:
                    cache_file.unlink()
                    logger.debug(f"Deleted expired cache file: {key}")
                except Exception as e:
                    logger.error(f"Failed to delete expired cache file {key}: {str(e)}")
                return None

            logger.debug(f"Cache hit: {key}")
            # Store in memory for faster subsequent access
            self.memory_cache.set(key, data["value"])
            return data["value"]

        except Exception as e:
            logger.error(f"Error reading cache file {key}: {str(e)}")
            try:
                cache_file.unlink()
                logger.info(f"Deleted corrupted cache file: {key}")
            except Exception as e2:
                logger.error(f"Failed to delete corrupted cache file {key}: {str(e2)}")
            return None

    def set(self, key: str, value: Any, cache_type: str = "dynamic") -> None:
        """
        Save data to disk cache with expiry.

        Args:
            key: The cache key
            value: The value to cache
            cache_type: Cache expiry type ('static', 'dynamic', or 'volatile')

        Raises:
            ValueError: If an invalid cache_type is provided
        """
        if cache_type not in CACHE_EXPIRY:
            raise ValueError(
                f"Invalid cache type: {cache_type}. Valid types: {list(CACHE_EXPIRY.keys())}"
            )

        # Store in memory cache
        self.memory_cache.set(key, value)

        cache_file = self.cache_dir / f"{key}.cache"
        temp_file = self.cache_dir / "temp" / f"{key}.cache.tmp"

        data = {
            "value": value,
            "expiry": datetime.now() + CACHE_EXPIRY[cache_type],
            "created": datetime.now().isoformat(),
        }

        try:
            # Write to temp file first
            temp_file.parent.mkdir(exist_ok=True)
            with open(temp_file, "wb") as f:
                pickle.dump(data, f)

            # Move temp file to final location
            shutil.move(str(temp_file), str(cache_file))
            logger.debug(f"Cached {key} with {cache_type} expiry")

        except Exception as e:
            logger.error(f"Error writing cache file {key}: {str(e)}")
            # Cleanup temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e2:
                logger.error(f"Failed to cleanup temp cache file: {str(e2)}")

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: The cache key

        Returns:
            bool: True if successfully deleted, False otherwise
        """
        self.memory_cache.delete(key)

        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.debug(f"Deleted cache file: {key}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete cache file {key}: {str(e)}")
                return False
        return False

    def clear(self, cache_type: Optional[str] = None) -> int:
        """
        Clear all cache entries or only those of a specific type.

        Args:
            cache_type: If provided, only clear caches of this type

        Returns:
            int: Number of cache entries cleared
        """
        self.memory_cache.clear()

        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    if cache_type:
                        # Open and check the type
                        with open(cache_file, "rb") as f:
                            data = pickle.load(f)
                            expiry = data["expiry"]
                            created = datetime.fromisoformat(data["created"])

                            # Calculate the timedelta between creation and expiry
                            delta = expiry - created

                            # Find matching cache type based on expiry delta
                            matching_type = None
                            for t, td in CACHE_EXPIRY.items():
                                if td == delta:
                                    matching_type = t
                                    break

                            if matching_type != cache_type:
                                continue

                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {str(e)}")

            logger.info(f"Cleared {count} cache entries")
            return count

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return count
