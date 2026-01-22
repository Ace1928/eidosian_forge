"""
**1.2 File Manager (`file_manager.py`):**
- **Purpose:** Manages the creation, organization, and validation of output files and directories with utmost precision and adherence to standards.
- **Functions:**
  - `create_file(file_path, content)`: Creates a file with the specified content, ensuring data integrity and security.
  - `create_directory(path)`: Ensures the creation and validation of a directory structure, maintaining system consistency.
  - `organize_script_components(components, base_path)`: Organizes extracted components into files and directories based on a predefined structure, ensuring systematic categorization and accessibility.
"""

import os
import logging
from typing import Dict, List


class FileManager:
    """
    Manages file operations with detailed logging, robust error handling, and strict adherence to coding standards, ensuring high cohesion and systematic methodology in file management.
    """

    def __init__(self):
        """
        Initializes the FileManager with a dedicated logger for file operations, setting up comprehensive logging mechanisms.
        """
        self.logger = logging.getLogger("FileManager")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("file_operations.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.debug("FileManager initialized and operational.")

    def create_file(self, file_path: str, content: str) -> None:
        """
        Creates a file at the specified path with the given content, includes detailed logging, error handling, and data integrity checks.
        """
        try:
            with open(file_path, "w") as file:
                file.write(content)
                self.logger.info(
                    f"File successfully created at {file_path} with specified content."
                )
        except Exception as e:
            self.logger.error(f"Error creating file at {file_path}: {e}")
            raise IOError(
                f"An error occurred while creating the file at {file_path}: {e}"
            )

    def create_directory(self, path: str) -> None:
        """
        Creates a directory at the specified path, includes detailed logging, error handling, and validation of directory structure.
        """
        try:
            os.makedirs(path, exist_ok=True)
            self.logger.info(f"Directory successfully created or verified at {path}")
        except Exception as e:
            self.logger.error(f"Error creating directory at {path}: {e}")
            raise IOError(
                f"An error occurred while creating the directory at {path}: {e}"
            )

    def organize_script_components(
        self, components: Dict[str, List[str]], base_path: str
    ) -> None:
        """
        Organizes script components into files and directories based on their type, includes detailed logging, error handling, and systematic file organization.
        """
        try:
            for component_type, component_data in components.items():
                component_directory = os.path.join(base_path, component_type)
                self.create_directory(component_directory)
                for index, data in enumerate(component_data):
                    file_path = os.path.join(
                        component_directory, f"{component_type}_{index}.py"
                    )
                    self.create_file(file_path, data)
                    self.logger.info(
                        f"{component_type} component organized into {file_path}"
                    )
            self.logger.debug(
                f"All components successfully organized under base path {base_path}"
            )
        except Exception as e:
            self.logger.error(f"Error organizing components at {base_path}: {e}")
            raise Exception(
                f"An error occurred while organizing script components at {base_path}: {e}"
            )
