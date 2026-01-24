"""
JSON Schema Validation
-------------------
Validates JSON data against a schema.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

from .cache import CacheManager

# Get logger
logger = logging.getLogger("eidos_validator.validator")


class SchemaValidator:
    """
    Validates JSON data against a schema with detailed error reporting.

    This class handles the validation of JSON data against a schema,
    with support for detailed error reporting, schema caching, and
    multiple validation modes.
    """

    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        """
        Initialize the schema validator.

        Args:
            schema_path: Path to the JSON schema file (Optional)
            cache_manager: Optional CacheManager instance to use for caching
        """
        self.schema_path = Path(schema_path) if schema_path else None
        self.cache_manager = cache_manager or CacheManager()
        self.available_modules: Dict[str, bool] = {}
        self._schema: Optional[Dict[str, Any]] = None

    def set_available_modules(self, modules: Dict[str, bool]) -> None:
        """
        Set the dictionary of available modules.

        Args:
            modules: Dictionary mapping module names to availability status
        """
        self.available_modules = modules

    def set_schema_path(self, schema_path: Union[str, Path]) -> None:
        """
        Set or update the path to the schema file.

        Args:
            schema_path: Path to the JSON schema file
        """
        self.schema_path = Path(schema_path)
        self._schema = None  # Reset cached schema

    def get_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the JSON schema from file or cache.

        Returns:
            Optional[Dict[str, Any]]: The JSON schema or None if not available

        Raises:
            FileNotFoundError: If schema file not found
            json.JSONDecodeError: If schema file contains invalid JSON
        """
        # Return cached schema if available
        if self._schema is not None:
            return self._schema

        # Try to get from disk cache
        schema_cache_key = "json_schema"
        schema = self.cache_manager.get(schema_cache_key)

        if schema:
            self._schema = schema
            logger.debug("Loaded schema from cache")
            return schema

        # Load from file if path is provided
        if self.schema_path and self.schema_path.exists():
            try:
                # Ensure utf-8 for emojis/symbols
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                self._schema = schema
                self.cache_manager.set(schema_cache_key, schema, "static")
                logger.info(f"Loaded and cached schema file from {self.schema_path}")
                return schema
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in schema file: {e}")
                raise
            except Exception as e:
                logger.error(f"Error reading schema file: {str(e)}")
                raise
        elif self.schema_path:
            logger.error(f"Schema file not found at {self.schema_path}")
            raise FileNotFoundError(f"Schema file not found at {self.schema_path}")
        else:
            logger.warning("No schema path provided")
            return None

    def validate(self, json_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate JSON data against the schema.

        Args:
            json_data: JSON data to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        schema = self.get_schema()

        if not schema:
            return (True, "Schema not available - validation skipped")

        if not self.available_modules.get("jsonschema", False):
            return (True, "jsonschema module not available - validation skipped")

        from jsonschema import validate, ValidationError

        try:
            validate(instance=json_data, schema=schema)
            return (True, "Validation successful! JSON conforms to schema.")
        except ValidationError as e:
            error_message = f"Validation error: {e.message}"
            path = "/".join(str(p) for p in e.path)
            if path:
                error_message += f" at path: {path}"
            logger.error(f"Schema validation failed: {error_message}")
            return (False, error_message)
        except Exception as e:
            error_message = f"Unexpected validation error: {str(e)}"
            logger.error(error_message)
            return (False, error_message)
