import json
import jsonschema
from typing import Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EidosianSchemaValidator:
    def __init__(self):
        """Initialize the schema validator."""
        self.schema = None
        self._load_schema()

    def _load_schema(self) -> None:
        """Load the Eidosian reflection schema."""
        try:
            schema_path = Path("eidosian_io_schema.json")
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found at {schema_path}")

            with open(schema_path, "r") as f:
                self.schema = json.load(f)

            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(self.schema)
            logger.info("Schema loaded and validated successfully")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load schema: {str(e)}")
            raise

    def validate_json(self, json_data: Dict[Any, Any]) -> bool:
        """
        Validate JSON data against the Eidosian reflection schema.

        Args:
            json_data: Dictionary containing the JSON data to validate

        Returns:
            bool: True if validation succeeds

        Raises:
            jsonschema.ValidationError: If validation fails
            jsonschema.SchemaError: If schema is invalid
            ValueError: If json_data is None
        """
        if json_data is None:
            raise ValueError("JSON data cannot be None")

        try:
            assert (
                self.schema is not None
            ), "Schema must be initialized before validation"
            jsonschema.validate(instance=json_data, schema=self.schema)
            logger.info("JSON validation successful")
            return True

        except jsonschema.ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            # Include path to the validation error
            logger.error(f"Failed validating {e.path}")
            raise
        except jsonschema.SchemaError as e:
            logger.error(f"Schema error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise

    def validate_json_file(self, file_path: str) -> bool:
        """
        Validate a JSON file against the Eidosian reflection schema.

        Args:
            file_path: Path to the JSON file to validate

        Returns:
            bool: True if validation succeeds

        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file contains invalid JSON
            jsonschema.ValidationError: If validation fails
            jsonschema.SchemaError: If schema is invalid
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "r") as f:
                json_data = json.load(f)
            return self.validate_json(json_data)

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            raise


def main():
    """Example usage of the schema validator."""
    try:
        # Initialize validator
        validator = EidosianSchemaValidator()

        # Example validation of a JSON file
        sample_json = {
            "version": "1.0.0",
            "metadata": {
                "creation_timestamp": "2025-01-05T12:00:00Z",
                "creator": "Eidos",
                "context": {"purpose": "Test validation", "scope": "Unit test"},
            },
            "storage_config": {
                "base_path": "/test/path",
                "file_structure": {
                    "cycles_dir": "/test/path/cycles",
                    "metadata_dir": "/test/path/metadata",
                    "synthesis_dir": "/test/path/synthesis",
                    "relationship_dir": "/test/path/relationships",
                },
            },
            "user_input": {"prompt": "Test prompt"},
            "reflection_cycles": [],
            "final_output": "Test complete",
        }

        # Validate the sample JSON
        is_valid = validator.validate_json(sample_json)
        logger.info(f"Validation result: {is_valid}")

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
