"""
Config Manager Module

This module offers a flexible and robust configuration manager designed to handle application settings across various modules and programs efficiently. It supports loading, retrieving, setting, and saving configuration values with ease, including type conversion and fallback support.

Author: Lloyd Handyside
Creation Date: 2024-04-08
Last Modified: 2024-04-08

Functionalities:
- Load and manage multiple configuration files.
- Retrieve and set configuration values with type conversion and fallback support.
- Save configuration changes back to files.
- Support for encryption of sensitive configuration values.
- Support for JSON-based configuration files.
"""

import configparser
import json
from cryptography.fernet import Fernet
from typing import Any, Dict, Optional, Union
import os
import logging


class ConfigManager:
    """
    A flexible and robust configuration manager to handle application settings across various modules and programs.

    Attributes:
        config_files (Dict[str, Union[configparser.ConfigParser, Dict]]): A dictionary mapping configuration file names to their parser instances or JSON dictionaries.
        encryption_key (bytes): The encryption key used for encrypting and decrypting sensitive configuration values.
        cipher_suite (Fernet): The cipher suite used for encryption and decryption.
    """

    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initializes the ConfigManager with an empty dictionary for config files and sets up encryption if an encryption key is provided.

        Args:
            encryption_key (Optional[bytes]): An optional encryption key for encrypting and decrypting sensitive values. A new key is generated if none is provided.
        """
        self.config_files: Dict[str, Union[configparser.ConfigParser, Dict]] = {}
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def load_config(
        self, file_path: str, config_name: Optional[str] = None, file_type: str = "ini"
    ) -> None:
        """
        Loads a configuration file into the manager, supporting both INI and JSON formats.

        Args:
            file_path (str): The path to the configuration file.
            config_name (Optional[str]): An optional name to reference the configuration file. Defaults to the file name if not provided.
            file_type (str): The type of the configuration file ('ini' or 'json').
        """
        if not config_name:
            config_name = os.path.basename(file_path)

        if file_type == "ini":
            config_parser = configparser.ConfigParser()
            config_parser.read(file_path)
            self.config_files[config_name] = config_parser
        elif file_type == "json":
            with open(file_path, "r") as json_file:
                config_data = json.load(json_file)
                self.config_files[config_name] = config_data
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def encrypt_value(self, value: str) -> str:
        """
        Encrypts a configuration value using the configured cipher suite.

        Args:
            value (str): The value to encrypt.

        Returns:
            str: The encrypted value.
        """
        return self.cipher_suite.encrypt(value.encode()).decode()

    def decrypt_value(self, value: str) -> str:
        """
        Decrypts a configuration value using the configured cipher suite.

        Args:
            value (str): The encrypted value to decrypt.

        Returns:
            str: The decrypted value.
        """
        return self.cipher_suite.decrypt(value.encode()).decode()

    def get(
        self,
        config_name: str,
        section: str,
        option: str,
        fallback: Optional[Any] = None,
        is_encrypted: bool = False,
    ) -> Union[str, int, float, bool, None]:
        """
        Retrieves a configuration value with support for fallbacks and optional decryption for sensitive values.

        Args:
            config_name (str): The name of the configuration file.
            section (str): The section in the configuration file (for INI files) or the top-level key (for JSON files).
            option (str): The option key to retrieve the value for.
            fallback (Optional[Any]): The fallback value if the option is not found.
            is_encrypted (bool): Specifies whether the value is encrypted and needs to be decrypted.

        Returns:
            Union[str, int, float, bool, None]: The value of the configuration option, converted to the appropriate type. If the value is encrypted, it is decrypted before conversion.
        """
        config_parser = self.config_files.get(config_name)
        if config_parser:
            if isinstance(
                config_parser, configparser.ConfigParser
            ) and config_parser.has_option(section, option):
                value = config_parser.get(section, option)
            elif (
                isinstance(config_parser, dict)
                and section in config_parser
                and option in config_parser[section]
            ):
                value = config_parser[section][option]
            else:
                return fallback

            if is_encrypted:
                value = self.decrypt_value(value)

            # Convert value to int, float, bool, or return as string
            for cast in (int, float):
                try:
                    return cast(value)
                except ValueError:
                    continue
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            return value
        return fallback

    def set(
        self,
        config_name: str,
        section: str,
        option: str,
        value: Any,
        is_encrypted: bool = False,
    ) -> None:
        """
        Sets a configuration value, with optional encryption for sensitive values.

        Args:
            config_name (str): The name of the configuration file.
            section (str): The section in the configuration file (for INI files) or the top-level key (for JSON files).
            option (str): The option key to set the value for.
            value (Any): The value to set.
            is_encrypted (bool): Specifies whether the value should be encrypted.
        """
        if is_encrypted:
            value = self.encrypt_value(str(value))

        config_parser = self.config_files.get(config_name)
        if not config_parser:
            raise ValueError(f"Configuration '{config_name}' not loaded.")
        if isinstance(config_parser, configparser.ConfigParser):
            if not config_parser.has_section(section):
                config_parser.add_section(section)
            config_parser.set(section, option, str(value))
        elif isinstance(config_parser, dict):
            if section not in config_parser:
                config_parser[section] = {}
            config_parser[section][option] = value
        else:
            raise ValueError(f"Unsupported configuration type for '{config_name}'.")

    def save_config(
        self, config_name: str, file_path: Optional[str] = None, file_type: str = "ini"
    ) -> None:
        """
        Saves the specified configuration back to a file, supporting both INI and JSON formats.

        Args:
            config_name (str): The name of the configuration to save.
            file_path (Optional[str]): The file path to save the configuration to. Uses the original path if not provided.
            file_type (str): The type of the configuration file ('ini' or 'json').
        """
        if not file_path:
            file_path = config_name  # Assuming config_name was the file path if no explicit file_path is provided
        config_parser = self.config_files.get(config_name)
        if not config_parser:
            raise ValueError(f"Configuration '{config_name}' not loaded.")
        if file_type == "ini" and isinstance(config_parser, configparser.ConfigParser):
            with open(file_path, "w") as configfile:
                config_parser.write(configfile)
        elif file_type == "json" and isinstance(config_parser, dict):
            with open(file_path, "w") as json_file:
                json.dump(config_parser, json_file, indent=4)
        else:
            raise ValueError(
                f"Unsupported configuration type or file type for '{config_name}'."
            )


__all__ = ["ConfigManager"]

# Example usage:
config_manager = ConfigManager()
config_manager.load_config("config.ini", "main_config")
print(
    config_manager.get(
        "main_config", "Database", "host", fallback="localhost", is_encrypted=False
    )
)
config_manager.set("main_config", "Database", "port", 5432, is_encrypted=False)
config_manager.save_config("main_config")

# TODO:
# - Explore adding support for YAML configuration files.
# - Consider implementing a caching mechanism for frequently accessed configurations.

# Known Issues:
# - JSON file saving does not pretty-print for nested objects.
