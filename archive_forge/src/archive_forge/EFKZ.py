"""
**1.5 Configuration Manager (`config_manager.py`):**
- **Purpose:** Manages external configuration settings with precision and systematic methodology.
- **Functions:**
  - `load_configuration_from_file(configuration_file_path, file_format='json')`: Dynamically loads configuration settings from a specified file format, either JSON or XML, with comprehensive error handling and logging.
"""

import json
import xml.etree.ElementTree as ET
import logging


class ConfigurationManager:
    def __init__(self):
        """
        Initializes the ConfigurationManager with a dedicated logger for tracking all operations related to configuration management.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("ConfigurationManager initialized successfully.")

    def load_configuration_from_file(
        self, configuration_file_path: str, file_format: str = "json"
    ) -> dict:
        """
        Loads configuration settings from a file specified by `configuration_file_path` in either JSON or XML format.

        Parameters:
            configuration_file_path (str): The file path to the configuration file.
            file_format (str): The format of the configuration file, either 'json' or 'xml'.

        Returns:
            dict: A dictionary containing the configuration settings.

        Raises:
            ValueError: If the specified file format is neither 'json' nor 'xml'.
            FileNotFoundError: If the configuration file does not exist.
            Exception: For any unforeseen errors during the loading process.

        Detailed logging is performed to ensure all steps are recorded for debugging and verification purposes.
        """
        try:
            self.logger.debug(
                f"Attempting to load configuration from {configuration_file_path} as {file_format}."
            )
            if file_format == "json":
                with open(configuration_file_path, "r") as file:
                    configuration = json.load(file)
                    self.logger.info("Configuration loaded successfully from JSON.")
            elif file_format == "xml":
                tree = ET.parse(configuration_file_path)
                root = tree.getroot()
                configuration = {child.tag: child.text for child in root}
                self.logger.info("Configuration loaded successfully from XML.")
            else:
                raise ValueError(
                    "Unsupported configuration file format specified. Use 'json' or 'xml'."
                )

            self.logger.debug(f"Configuration Data: {configuration}")
            return configuration
        except FileNotFoundError:
            self.logger.error(
                f"The configuration file at {configuration_file_path} was not found."
            )
            raise
        except ValueError as ve:
            self.logger.error(f"Value error occurred: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise
