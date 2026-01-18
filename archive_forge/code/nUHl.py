class ScriptParser:
    """
    A class dedicated to parsing Python scripts with comprehensive logging and parsing capabilities.
    This class adheres to high standards of modularity, ensuring each method serves a single focused purpose.
    """

    def __init__(self, script_content: str):
        """
        Initialize the PythonScriptParser with the script content and a dedicated logger.

        Parameters:
            script_content (str): The content of the Python script to be parsed.
        """
        self.script_content = script_content
        self.parser_logger = logging.getLogger(__name__)
        self.parser_logger.debug(
            "PythonScriptParser initialized with provided script content."
        )

    def extract_import_statements(self) -> list:
        """
        Extracts import statements using regex with detailed logging.

        Returns:
            list: A list of import statements extracted from the script content.
        """
        self.parser_logger.debug("Attempting to extract import statements.")
        import_statements = re.findall(
            r"^\s*import .*", self.script_content, re.MULTILINE
        )
        self.parser_logger.info(
            f"Extracted {len(import_statements)} import statements."
        )
        return import_statements

    def extract_documentation(self) -> list:
        """
        Extracts block and inline documentation with detailed logging.

        Returns:
            list: A list of documentation blocks and inline comments extracted from the script content.
        """
        self.parser_logger.debug(
            "Attempting to extract documentation blocks and inline comments."
        )
        documentation_blocks = re.findall(
            r'""".*?"""|\'\'\'.*?\'\'\'|#.*$',
            self.script_content,
            re.MULTILINE | re.DOTALL,
        )
        self.parser_logger.info(
            f"Extracted {len(documentation_blocks)} documentation blocks."
        )
        return documentation_blocks

    def extract_class_definitions(self) -> list:
        """
        Uses AST to extract class definitions with detailed logging.

        Returns:
            list: A list of class definitions extracted from the script content using AST.
        """
        self.parser_logger.debug("Attempting to extract class definitions using AST.")
        tree = ast.parse(self.script_content)
        class_definitions = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        self.parser_logger.info(
            f"Extracted {len(class_definitions)} class definitions."
        )
        return class_definitions

    def extract_function_definitions(self) -> list:
        """
        Uses AST to extract function definitions with detailed logging.

        Returns:
            list: A list of function definitions extracted from the script content using AST.
        """
        self.parser_logger.debug(
            "Attempting to extract function definitions using AST."
        )
        tree = ast.parse(self.script_content)
        function_definitions = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        self.parser_logger.info(
            f"Extracted {len(function_definitions)} function definitions."
        )
        return function_definitions

    def identify_main_executable_block(self) -> list:
        """
        Identifies the main executable block of the script with detailed logging.

        Returns:
            list: A list containing the main executable block of the script.
        """
        self.parser_logger.debug("Attempting to identify the main executable block.")
        main_executable_block = re.findall(
            r'if __name__ == "__main__":\s*(.*)', self.script_content, re.DOTALL
        )
        self.parser_logger.info("Main executable block identified.")
        return main_executable_block
