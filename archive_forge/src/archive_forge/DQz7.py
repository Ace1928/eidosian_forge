"""
**1.2 File Manager (`file_manager.py`):**
- **Purpose:** Manages the creation and organization of output files and directories.
- **Functions:**
  - `create_file(file_path, content)`: Creates a file with the specified content.
  - `create_directory(path)`: Ensures the creation of a directory structure.
  - `organize_script_components(components, base_path)`: Organizes extracted components into files and directories based on a predefined structure.
"""


class FileOperationsManager:
    """
    Manages file operations with detailed logging and robust error handling, ensuring high cohesion and strict adherence to coding standards.
    """

    def __init__(self):
        """
        Initializes the FileOperationsManager with a dedicated logger for file operations.
        """
        self.file_operations_logger = logging.getLogger("FileOperationsManager")
        self.file_operations_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("file_operations.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.file_operations_logger.addHandler(handler)
        self.file_operations_logger.debug(
            "FileOperationsManager initialized and operational."
        )

    def create_file(self, file_path: str, content: str):
        """
        Creates a file at the specified path with the given content, includes detailed logging and error handling.
        """
        try:
            with open(file_path, "w") as file:
                file.write(content)
                self.file_operations_logger.info(
                    f"File successfully created at {file_path} with specified content."
                )
        except Exception as e:
            self.file_operations_logger.error(
                f"Error creating file at {file_path}: {e}"
            )
            raise IOError(
                f"An error occurred while creating the file at {file_path}: {e}"
            )

    def create_directory(self, path: str):
        """
        Creates a directory at the specified path, includes detailed logging and error handling.
        """
        try:
            os.makedirs(path, exist_ok=True)
            self.file_operations_logger.info(
                f"Directory successfully created or verified at {path}"
            )
        except Exception as e:
            self.file_operations_logger.error(
                f"Error creating directory at {path}: {e}"
            )
            raise IOError(
                f"An error occurred while creating the directory at {path}: {e}"
            )

    def organize_script_components(self, components: dict, base_path: str):
        """
        Organizes script components into files and directories based on their type, includes detailed logging and error handling.
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
                    self.file_operations_logger.info(
                        f"{component_type} component organized into {file_path}"
                    )
            self.file_operations_logger.debug(
                f"All components successfully organized under base path {base_path}"
            )
        except Exception as e:
            self.file_operations_logger.error(
                f"Error organizing components at {base_path}: {e}"
            )
            raise Exception(
                f"An error occurred while organizing script components at {base_path}: {e}"
            )
