"""
This module provides functionality to convert markdown files or directories containing markdown files into JSONL format. 
It includes classes for parsing markdown content, converting it to JSONL, and processing directories of markdown files.
"""

import json
import os
import logging
import argparse
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "LoggerConfigurator",
    "MarkdownParser",
    "JSONLConverter",
    "MarkdownProcessor",
    "CmdArgumentParser",
    "TestMarkdownConversion",
]


class LoggerConfigurator:
    """Configures and manages logging for the application.

    This class encapsulates the logging setup process, allowing for easy configuration
    of logging both to file and console based on the provided arguments.

    Attributes:
        enable_console_logging (bool): Determines if logging to console is enabled.
    """

    _instance = None

    def __new__(cls, enable_console_logging: bool = True):
        if cls._instance is None:
            cls._instance = super(LoggerConfigurator, cls).__new__(cls)
            cls._instance.enable_console_logging = enable_console_logging
            cls._instance._configure_logging()
        return cls._instance

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
    """Parses markdown content to extract conversations."""

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

        USER_PREFIX = "##USER##"
        ASSISTANT_PREFIX = "##ASSISTANT##"

        for line in file_content.split("\n"):
            line = line.strip()
            if line.startswith(USER_PREFIX):
                if (
                    user_input and assistant_response
                ):  # Save previous conversation before starting a new user input
                    conversations.append(
                        {"input": user_input, "output": assistant_response}
                    )
                    user_input, assistant_response = (
                        "",
                        "",
                    )  # Reset for the next conversation
                user_input += " " + line[len(USER_PREFIX) :].strip()
                capture_mode = "user"
            elif line.startswith(ASSISTANT_PREFIX):
                assistant_response += " " + line[len(ASSISTANT_PREFIX) :].strip()
                if (
                    capture_mode != "assistant"
                ):  # Switch to assistant mode if not already
                    capture_mode = "assistant"
                if not user_input:  # Handle consecutive assistant responses
                    logger.warning(
                        "Consecutive assistant response found. Concatenating."
                    )
            else:
                if capture_mode == "user":
                    user_input += " " + line
                elif capture_mode == "assistant":
                    assistant_response += " " + line

        # Check for and add the last conversation if it exists
        if (
            user_input or assistant_response
        ):  # Allow for assistant response without user input
            conversations.append(
                {"input": user_input.strip(), "output": assistant_response.strip()}
            )

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
            with open(output_path, "w", encoding="utf-8") as f:
                for conversation in conversations:
                    f.write(json.dumps(conversation) + "\n")
            logger.info(f"Conversations successfully written to {output_path}")
        except FileNotFoundError as e:
            logger.error(
                f"Failed to write conversations to JSONL due to {e}", exc_info=True
            )


class MarkdownProcessor:
    """Processes markdown files or directories to convert them to JSONL format."""

    total_conversations = (
        0  # Class variable to keep track of total conversations across all directories
    )

    @staticmethod
    def process_directory(
        input_dir: str, output_dir: str, index_file_path: str
    ) -> None:
        """Processes all markdown files in a directory and converts them to JSONL format.

        Args:
            input_dir (str): The directory containing markdown files.
            output_dir (str): The directory where JSONL files will be saved.
            index_file_path (str): The path to the index file recording all conversations.
        """
        logger = logging.getLogger("MarkdownProcessor")
        os.makedirs(output_dir, exist_ok=True)
        directory_conversations = (
            0  # To keep track of conversations in the current directory
        )

        with ThreadPoolExecutor() as executor, open(
            index_file_path, "a", encoding="utf-8"
        ) as index_file:
            futures = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".md"):
                        md_path = os.path.join(root, file)
                        futures.append(
                            executor.submit(
                                MarkdownProcessor.process_file,
                                md_path,
                                output_dir,
                                index_file,
                            )
                        )

            for future in futures:
                directory_conversations += (
                    future.result()
                )  # Assuming process_file returns the number of conversations processed

            MarkdownProcessor.total_conversations += directory_conversations
            logger.info(
                f"Processed {directory_conversations} conversations in {input_dir}."
            )

        # Append the total count for the directory to the index file
        index_file.write(
            f"Total conversations in {input_dir}: {directory_conversations}\n"
        )

    @staticmethod
    def finalize_index_file(index_file_path: str) -> None:
        """Finalizes the index file with the total conversation count.

        Args:
            index_file_path (str): The path to the index file.
        """
        with open(index_file_path, "a", encoding="utf-8") as index_file:
            index_file.write(
                f"Total conversations processed: {MarkdownProcessor.total_conversations}\n"
            )

    @staticmethod
    def process_file(md_path: str, output_dir: str, index_file) -> int:
        """Processes a single markdown file and converts it to JSONL format, also updates the index file.

        Args:
            md_path (str): The path to the markdown file.
            output_dir (str): The directory where the JSONL file will be saved.
            index_file (_io.TextIOWrapper): The open file object for the index file.

        Returns:
            int: The number of conversations processed in the markdown file.
        """
        logger = logging.getLogger("MarkdownProcessor")
        output_path = Path(output_dir) / (Path(md_path).stem + ".jsonl")
        conversations_count = 0

        try:
            with open(md_path, "r", encoding="utf-8") as md_file:
                file_content = md_file.read()
                conversations = MarkdownParser.parse_content(file_content)
                JSONLConverter.convert(conversations, str(output_path))
                conversations_count = len(conversations)
                # Write the processed file and its conversation count to the index file
                index_file.write(f"{md_path}: {conversations_count} conversations\n")
        except FileNotFoundError as e:
            logger.error(f"Markdown file not found: {e}", exc_info=True)

        return conversations_count


class CmdArgumentParser:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(
            description="Convert Markdown directories to JSONL format."
        )
        parser.add_argument(
            "--input_dir",
            required=True,
            help="The input directory containing markdown files.",
        )
        parser.add_argument(
            "--output_dir", required=True, help="The output directory for JSONL files."
        )
        return parser.parse_args()


if __name__ == "__main__":
    LoggerConfigurator(enable_console_logging=True)
    args = CmdArgumentParser.parse()
    input_directory = args.input_dir
    output_directory = args.output_dir
    index_file_path = Path(output_directory) / "index.txt"

    try:
        MarkdownProcessor.process_directory(
            input_directory, output_directory, str(index_file_path)
        )
        MarkdownProcessor.finalize_index_file(str(index_file_path))
        logger = logging.getLogger("main")
        logger.info("Markdown conversion process completed successfully.")
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during the Markdown conversion process.",
            exc_info=True,
        )
