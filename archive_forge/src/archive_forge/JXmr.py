import json
import os
import logging
import argparse
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod

__all__ = [
    "LoggerConfigurator",
    "MarkdownParser",
    "JSONLConverter",
    "MarkdownProcessor",
    "ArgumentParser",
    "TestMarkdownConversion",
]


class LoggerConfigurator:
    """Configures and manages logging for the application.

    This class encapsulates the logging setup process, allowing for easy configuration
    of logging both to file and console based on the provided arguments.

    Attributes:
        enable_console_logging (bool): Determines if logging to console is enabled.
    """

    def __init__(self, enable_console_logging: bool = True) -> None:
        """Initializes the LoggerConfigurator with optional console logging.

        Args:
            enable_console_logging (bool): Flag to enable logging to the console. Defaults to True.
        """
        self.enable_console_logging = enable_console_logging
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configures the logging settings for the application."""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename="md_to_jsonl_conversion.log",
            filemode="a",
        )
        if self.enable_console_logging:
            self._enable_console_logging()

    def _enable_console_logging(self) -> None:
        """Enables logging output to the console."""
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)


class MarkdownParser:
    """Parses markdown content to extract conversations.

    This class is responsible for parsing markdown files, identifying conversation
    segments, and extracting these segments into a structured format.

    Methods:
        parse_content(file_content: str) -> List[Dict[str, str]]: Parses the markdown content and returns conversations.
    """

    @staticmethod
    def parse_content(file_content: str) -> List[Dict[str, str]]:
        """Parses markdown content and extracts conversations.

        Args:
            file_content (str): The content of the markdown file as a string.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the parsed conversations.
        """
        logger = logging.getLogger("MarkdownParser")
        conversations = []
        user_input, assistant_response = "", ""
        capture_mode: Optional[str] = None

        for line in file_content.split("\n"):
            if line.startswith("## USER"):
                if assistant_response:
                    conversations.append(
                        {
                            "input": user_input.strip(),
                            "output": assistant_response.strip(),
                        }
                    )
                    user_input, assistant_response = "", ""
                capture_mode = "user"
            elif line.startswith("## ASSISTANT"):
                capture_mode = "assistant"
            elif capture_mode == "user":
                user_input += line.strip() + " "
            elif capture_mode == "assistant":
                assistant_response += line.strip() + " "

        if user_input and assistant_response:
            conversations.append(
                {"input": user_input.strip(), "output": assistant_response.strip()}
            )

        # Filter out empty conversation pairs
        conversations = [
            conv for conv in conversations if conv["input"] and conv["output"]
        ]
        logger.info(
            f"Extracted {len(conversations)} conversations from Markdown content."
        )
        return conversations


class JSONLConverter:
    """Converts conversations to JSONL format and writes them to a file.

    This class handles the conversion of structured conversation data into the JSONL format,
    which is then written to a specified output file.

    Methods:
        convert(conversations: List[Dict[str, str]], output_path: str) -> None: Converts and writes conversations to a file.
    """

    @staticmethod
    def convert(conversations: List[Dict[str, str]], output_path: str) -> None:
        """Converts conversations to JSONL format and writes them to a file.

        Args:
            conversations (List[Dict[str, str]]): The conversations to be converted.
            output_path (str): The path to the output file.
        """
        logger = logging.getLogger("JSONLConverter")
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                for conversation in conversations:
                    f.write(json.dumps(conversation) + "\n")
            logger.info(f"Conversations successfully written to {output_path}")
        except FileNotFoundError as e:
            logger.error(
                f"Failed to write conversations to JSONL due to {e}", exc_info=True
            )


class MarkdownProcessor:
    """Processes markdown files or directories to convert them to JSONL format.

    This class provides functionality to process individual markdown files or entire directories
    containing markdown files, converting the extracted conversations to JSONL format.

    Methods:
        process_file(md_path: str, output_dir: str) -> None: Processes a single markdown file.
        process_directory(input_dir: str, output_dir: str) -> None: Processes all markdown files in a directory.
    """

    @staticmethod
    def process_file(md_path: str, output_dir: str) -> None:
        """Processes a single markdown file and converts it to JSONL format.

        Args:
            md_path (str): The path to the markdown file.
            output_dir (str): The directory where the JSONL file will be saved.
        """
        logger = logging.getLogger("MarkdownProcessor")
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(md_path))[0] + ".jsonl"
        )

        try:
            with open(md_path, "r", encoding="utf-8") as md_file:
                file_content = md_file.read()
                conversations = MarkdownParser.parse_content(file_content)
                JSONLConverter.convert(conversations, output_path)
        except FileNotFoundError as e:
            logger.error(f"Markdown file not found: {e}", exc_info=True)

    @staticmethod
    def process_directory(input_dir: str, output_dir: str) -> None:
        """Processes all markdown files in a directory and converts them to JSONL format.

        Args:
            input_dir (str): The directory containing markdown files.
            output_dir (str): The directory where JSONL files will be saved.
        """
        logger = logging.getLogger("MarkdownProcessor")
        os.makedirs(output_dir, exist_ok=True)

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".md"):
                    md_path = os.path.join(root, file)
                    MarkdownProcessor.process_file(md_path, output_dir)

        logger.info(f"All Markdown files in {input_dir} have been processed.")


class ArgumentParser:
    """Parses command-line arguments for the script.

    This class encapsulates the command-line argument parsing functionality, providing
    a structured way to access the arguments required for the script to run.

    Methods:
        parse() -> argparse.Namespace: Parses and returns the command-line arguments.
    """

    @staticmethod
    def parse() -> argparse.Namespace:
        """Parses command-line arguments for the script.

        Returns:
            argparse.Namespace: The parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Convert Markdown files to JSONL format."
        )
        parser.add_argument(
            "--input_dir", required=True, help="Directory containing Markdown files."
        )
        parser.add_argument(
            "--output_dir", required=True, help="Directory to save JSONL files."
        )
        return parser.parse_args()


# The main execution block and test cases are omitted for brevity but would follow the same principles of encapsulation and abstraction.
