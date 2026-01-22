"""
Markdown to JSONL Converter Module
==================================

This module provides functionality to convert markdown files or directories containing markdown files into JSONL format.
It is designed to facilitate the processing of conversational data stored in markdown format by extracting conversations
and converting them into a more structured JSONL format. This can be particularly useful for NLP tasks and data analysis.

Main Components:
- LoggerConfigurator: Manages application-wide logging configuration.
- MarkdownParser: Parses markdown files to extract conversational data.
- JSONLConverter: Converts extracted conversations into JSONL format.
- MarkdownProcessor: Orchestrates the processing of directories containing markdown files.

Author: CursorBot
Creation Date: YYYY-MM-DD

Usage:
The module is designed to be run as a standalone script with input and output directory paths provided via command-line arguments.
"""

import json
import os
import logging
import click
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import markdown2
from tqdm import tqdm

__all__ = [
    "LoggerConfigurator",
    "MarkdownParser",
    "JSONLConverter",
    "MarkdownProcessor",
]


def log_method_call(method):
    """
    A decorator for logging method calls.

    Args:
        method (Callable): The method to be decorated.

    Returns:
        Callable: The wrapped method with logging functionality.
    """

    def wrapper(*args, **kwargs):
        logger = logging.getLogger(method.__qualname__)
        logger.info(f"Entering {method.__name__}")
        result = method(*args, **kwargs)
        logger.info(f"Exiting {method.__name__}")
        return result

    return wrapper


class LoggerConfigurator:
    """
    Configures and manages logging for the application.

    This class encapsulates the logging setup process, allowing for easy configuration
    of logging both to file and console based on the provided arguments. It follows the Singleton pattern
    to ensure only one instance manages the logging configuration throughout the application lifecycle.

    Attributes:
        enable_console_logging (bool): Determines if logging to console is enabled.

    Args:
        enable_console_logging (bool): If True, enables logging to the console. Defaults to True.
    """

    enable_console_logging: bool = True
    _instance = None

    def __new__(cls, enable_console_logging: bool = True):
        """
        Ensures that only one instance of LoggerConfigurator is created (Singleton pattern).
        Configures logging upon the first instantiation.

        Args:
            enable_console_logging (bool): If True, enables logging to the console. Defaults to True.
        """
        if cls._instance is None:
            cls._instance = super(LoggerConfigurator, cls).__new__(cls)
            cls._instance.enable_console_logging = enable_console_logging
            cls._instance._configure_logging()
        return cls._instance

    def _configure_logging(self) -> None:
        """Configures the logging settings for the application."""
        logging.basicConfig(
            level=logging.INFO,
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
    """
    Parses markdown content to extract conversations.

    This class is responsible for parsing markdown files, identifying and extracting conversational data
    structured as user inputs and assistant responses. The extracted conversations are then formatted into
    a list of dictionaries, facilitating their conversion into JSONL format by the JSONLConverter class.

    Methods:
        parse_content: Parses the content of a markdown file and extracts conversations.
    """

    @staticmethod
    @log_method_call
    def parse_content(file_content: str) -> List[Dict[str, str]]:
        """
        Parses markdown content and extracts conversations.

        This method iterates through each line of the provided markdown content,
        identifying user inputs and assistant responses based on predefined prefixes.
        Conversations are captured and stored in a list of dictionaries.

        Args:
            file_content (str): The content of the markdown file as a string.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the parsed conversations.
        """
        logger = logging.getLogger("MarkdownParser")
        conversations = []
        user_input, assistant_response = "", ""
        capture_mode: Optional[str] = None

        USER_PREFIX = "## USER"
        ASSISTANT_PREFIX = "## ASSISTANT"

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
    """
    Converts conversations to JSONL format and writes them to a file.

    This class is tasked with converting structured conversation data, provided as a list of dictionaries,
    into the JSONL format. The conversion results are then written to a specified output file. This format
    is particularly useful for machine learning and NLP tasks, as it allows for easy ingestion of conversational
    data into various processing pipelines.

    Methods:
        convert: Converts conversations into JSONL format and writes them to an output file.
    """

    @staticmethod
    @log_method_call
    def convert(conversations: List[Dict[str, str]], output_path: str) -> None:
        """
        Converts conversations to JSONL format and writes them to a file.

        This method iterates over a list of conversation dictionaries, converting each into a JSONL string.
        These strings are then written to the specified output file, creating a dataset ready for further
        processing or analysis.

        Args:
            conversations (List[Dict[str, str]]): The conversations to be converted.
            output_path (str): The path to the output file.

        Raises:
            FileNotFoundError: If the specified output file path does not exist.
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
    """
    Processes markdown files or directories to convert them to JSONL format.

    This class orchestrates the conversion of markdown files, or entire directories of markdown files,
    into the JSONL format. It leverages the MarkdownParser to extract conversational data from markdown
    files and the JSONLConverter to format and write this data to output files. Additionally, it maintains
    a count of all processed conversations, providing insights into the volume of data processed.

    Methods:
        process_directory: Processes all markdown files in a directory, converting them to JSONL.
        finalize_index_file: Finalizes the index file with the total conversation count.
        process_file: Processes a single markdown file, converting it to JSONL and updating the index file.
    """

    total_conversations = (
        0  # Class variable to keep track of total conversations across all directories
    )

    @staticmethod
    @log_method_call
    def process_directory(
        input_dir: str, output_dir: str, index_file_path: str
    ) -> None:
        """
        Processes all markdown files in a directory and converts them to JSONL format.

        This method scans the specified input directory for markdown files, processing each file
        found by extracting conversations and converting them to JSONL format. The results are saved
        in the specified output directory. Additionally, an index file is updated with the count of
        conversations processed for each file, providing a summary of the operation.

        Args:
            input_dir (str): The directory containing markdown files.
            output_dir (str): The directory where JSONL files will be saved.
            index_file_path (str): The path to the index file recording all conversations.

        Raises:
            Exception: If an error occurs during the processing of markdown files.
        """
        logger = logging.getLogger("MarkdownProcessor")
        os.makedirs(output_dir, exist_ok=True)
        directory_conversations = (
            0  # To keep track of conversations in the current directory
        )

        files_to_process = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(input_dir)
            for file in files
            if file.endswith(".md")
        ]
        num_workers = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() is None
        with ThreadPoolExecutor(max_workers=num_workers) as executor, tqdm(
            total=len(files_to_process), desc="Processing files"
        ) as progress:
            future_to_file = {
                executor.submit(
                    MarkdownProcessor.process_file, md_path, output_dir, index_file_path
                ): md_path
                for md_path in files_to_process
            }
            for future in as_completed(future_to_file):
                md_path = future_to_file[future]
                try:
                    directory_conversations += future.result()
                    progress.update(1)
                except Exception as exc:
                    logger.error(f"{md_path} generated an exception: {exc}")

            MarkdownProcessor.total_conversations += directory_conversations
            logger.info(
                f"Processed {directory_conversations} conversations in {input_dir}."
            )

        # Append the total count for the directory to the index file
        with open(index_file_path, "a", encoding="utf-8") as index_file:
            index_file.write(
                f"Total conversations in {input_dir}: {directory_conversations}\n"
            )

    @staticmethod
    @log_method_call
    def finalize_index_file(index_file_path: str) -> None:
        """
        Finalizes the index file with the total conversation count.

        After processing all markdown files, this method appends the total count of conversations
        processed to the index file. This provides a comprehensive summary of the conversion operation,
        detailing the total volume of conversational data processed.

        Args:
            index_file_path (str): The path to the index file.
        """
        with open(index_file_path, "a", encoding="utf-8") as index_file:
            index_file.write(
                f"Total conversations processed: {MarkdownProcessor.total_conversations}\n"
            )

    @staticmethod
    @log_method_call
    def process_file(md_path: str, output_dir: str, index_file_path: str) -> int:
        """
        Processes a single markdown file and converts it to JSONL format, also updates the index file.

        This method reads a markdown file, extracts conversational data using the MarkdownParser, and
        converts this data into JSONL format using the JSONLConverter. The resulting JSONL file is saved
        in the specified output directory. Additionally, the index file is updated with the count of
        conversations processed for the file, providing a record of the operation.

        Args:
            md_path (str): The path to the markdown file.
            output_dir (str): The directory where the JSONL file will be saved.
            index_file_path (str): The path to the index file.

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
                with open(index_file_path, "a", encoding="utf-8") as index_file:
                    index_file.write(
                        f"{md_path}: {conversations_count} conversations\n"
                    )
        except FileNotFoundError as e:
            logger.error(f"Markdown file not found: {e}", exc_info=True)

        return conversations_count


@click.command()
@click.option(
    "--input_dir",
    prompt="Input directory",
    help="The input directory containing markdown files.",
)
@click.option(
    "--output_dir",
    prompt="Output directory",
    help="The output directory for JSONL files.",
)
def main(input_dir, output_dir):
    """
    The main entry point of the script when run as a standalone module.

    This function gathers input and output directory paths via command-line arguments,
    initializes the logging configuration, and triggers the markdown processing operation.
    It orchestrates the conversion of markdown files within the specified directory to JSONL format,
    leveraging the MarkdownProcessor class to handle the bulk of the processing work.

    Args:
        input_dir (str): The input directory containing markdown files to be processed.
        output_dir (str): The output directory where JSONL files will be saved.
    """
    logger = logging.getLogger(__name__)
    LoggerConfigurator(enable_console_logging=True)
    index_file_path = Path(output_dir) / "index.txt"

    try:
        MarkdownProcessor.process_directory(input_dir, output_dir, str(index_file_path))
        MarkdownProcessor.finalize_index_file(str(index_file_path))
        logger.info("Markdown conversion process completed successfully.")
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during the Markdown conversion process.",
            exc_info=True,
        )


if __name__ == "__main__":
    main()

# TODO List for Future Improvements:
# - Implement a more sophisticated logging mechanism that allows for different logging levels to be set via command-line arguments.
# - Enhance the MarkdownParser to support more complex markdown structures and nested conversations.
# - Add functionality to JSONLConverter for validating the structure of the converted conversations against a predefined schema.
# - Explore parallel processing optimizations in MarkdownProcessor to improve performance on large directories.
# - Introduce a configuration file for setting up common parameters and preferences, reducing the reliance on hard-coded values.
# - Develop a GUI interface for users who prefer not to use command-line tools, providing a more accessible option for markdown conversion.
# - Investigate and handle potential edge cases and errors more robustly, ensuring the tool is resilient against malformed markdown files.
# - Expand the tool's capabilities to include the conversion of markdown to other formats, such as CSV or XML, for users with different needs.

# Known Issues:
# - Large markdown files with complex structures may slow down the parsing process, requiring optimization.
# - The current error handling mechanism may not catch all exceptions, potentially leading to uninformative error messages for the end-user.
# - Dependency on external libraries like markdown2 and tqdm may introduce compatibility issues with future updates of these libraries.
