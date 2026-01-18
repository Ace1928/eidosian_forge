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


if __name__ == "__main__":
    setup_logger()
    args = parse_arguments()
    input_directory = args.input_dir
    output_directory = args.output_dir

    try:
        process_md_directory(input_directory, output_directory)
        logger = logging.getLogger("main")
        logger.info("Markdown conversion process completed successfully.")
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during the Markdown conversion process.",
            exc_info=True,
        )

import unittest


class TesMarkdownConverstion(unittest.TestCase):
    def test_parse_md_content(self):
        file_content = """
        ## USER
        Hello!
        ## ASSISTANT
        Hi there!
        ## USER
        How are you?
        ## ASSISTANT
        I'm doing great, thanks for asking!
        """
        conversations = parse_md_content(file_content)
        self.assertEqual(len(conversations), 2)  # Two conversations extracted
        self.assertEqual(
            conversations[0]["input"], "Hello!"
        )  # First conversation input
        self.assertEqual(
            conversations[0]["output"], "Hi there!"
        )  # First conversation output
        self.assertEqual(
            conversations[1]["input"], "How are you?"
        )  # Second conversation input
        self.assertEqual(
            conversations[1]["output"], "I'm doing great, thanks for asking!"
        )  # Second conversation output

        def test_empty_md_content(self):
            """Test parsing of empty markdown content."""
            file_content = ""
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations), 0, "Expected no conversations for empty content"
            )

        def test_incomplete_conversation(self):
            """Test parsing of markdown content with incomplete conversation."""
            file_content = """
            ## USER
            Hello!
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations),
                0,
                "Expected no conversations for incomplete content",
            )

        def test_multiple_user_inputs(self):
            """Test parsing of markdown content with multiple user inputs before an assistant response."""
            file_content = """
            ## USER
            Hello!
            ## USER
            Are you there?
            ## ASSISTANT
            Yes, I'm here!
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(len(conversations), 1, "Expected one conversation")
            self.assertEqual(
                conversations[0]["input"],
                "Hello! Are you there?",
                "Expected concatenated user inputs",
            )

        def test_multiple_assistant_responses(self):
            """Test parsing of markdown content with multiple assistant responses before a user input."""
            file_content = """
            ## USER
            Hello!
            ## ASSISTANT
            Hi!
            ## ASSISTANT
            How can I help you?
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(len(conversations), 1, "Expected one conversation")
            self.assertEqual(
                conversations[0]["output"],
                "Hi! How can I help you?",
                "Expected concatenated assistant responses",
            )

        def test_no_user_input(self):
            """Test parsing of markdown content with no user input."""
            file_content = """
            ## ASSISTANT
            Welcome!
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations),
                0,
                "Expected no conversations for missing user input",
            )

        def test_no_assistant_response(self):
            """Test parsing of markdown content with no assistant response."""
            file_content = """
            ## USER
            Hello!
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations),
                0,
                "Expected no conversations for missing assistant response",
            )

        def test_whitespace_handling(self):
            """Test parsing of markdown content with excessive whitespace."""
            file_content = """
            ## USER
            
            Hello!   
            
            ## ASSISTANT
            
            Hi there!    
            
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(len(conversations), 1, "Expected one conversation")
            self.assertEqual(
                conversations[0]["input"], "Hello!", "Expected trimmed user input"
            )
            self.assertEqual(
                conversations[0]["output"],
                "Hi there!",
                "Expected trimmed assistant response",
            )

        def test_real_world_example(self):
            """Test parsing of a more complex real-world example."""
            file_content = """
            ## USER
            What's the weather like today?
            ## ASSISTANT
            It's sunny and warm.
            ## USER
            Great, should I wear sunscreen?
            ## ASSISTANT
            Yes, it's always a good idea to protect your skin.
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(len(conversations), 2, "Expected two conversations")
            self.assertEqual(
                conversations[0]["input"],
                "What's the weather like today?",
                "First conversation input mismatch",
            )
            self.assertEqual(
                conversations[0]["output"],
                "It's sunny and warm.",
                "First conversation output mismatch",
            )
            self.assertEqual(
                conversations[1]["input"],
                "Great, should I wear sunscreen?",
                "Second conversation input mismatch",
            )
            self.assertEqual(
                conversations[1]["output"],
                "Yes, it's always a good idea to protect your skin.",
                "Second conversation output mismatch",
            )

        def test_mixed_content_handling(self):
            """Test parsing of markdown content with mixed non-conversation lines."""
            file_content = """
            Random introduction text not part of the conversation.
            ## USER
            What's the time?
            Some interjection text.
            ## ASSISTANT
            It's 3 PM.
            Closing remarks not part of the conversation.
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations), 1, "Expected one conversation despite mixed content"
            )
            self.assertEqual(
                conversations[0]["input"],
                "What's the time?",
                "Input mismatch in mixed content",
            )
            self.assertEqual(
                conversations[0]["output"],
                "It's 3 PM.",
                "Output mismatch in mixed content",
            )

        def test_consecutive_conversations(self):
            """Test parsing of markdown content with consecutive conversations without breaks."""
            file_content = """
            ## USER
            What's your name?
            ## ASSISTANT
            I'm an AI assistant.
            ## USER
            Can you help me with Python?
            ## ASSISTANT
            Sure, what do you need help with?
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations), 2, "Expected two separate conversations"
            )
            self.assertEqual(
                conversations[0]["input"],
                "What's your name?",
                "First conversation input mismatch",
            )
            self.assertEqual(
                conversations[0]["output"],
                "I'm an AI assistant.",
                "First conversation output mismatch",
            )
            self.assertEqual(
                conversations[1]["input"],
                "Can you help me with Python?",
                "Second conversation input mismatch",
            )
            self.assertEqual(
                conversations[1]["output"],
                "Sure, what do you need help with?",
                "Second conversation output mismatch",
            )

        def test_invalid_format_handling(self):
            """Test parsing of markdown content with invalid formatting."""
            file_content = """
            USER
            Is this valid?
            ASSISTANT
            No, this is not correctly formatted.
            """
            conversations = parse_md_content(file_content)
            self.assertEqual(
                len(conversations),
                0,
                "Expected no conversations due to invalid formatting",
            )


# The main execution block and test cases are omitted for brevity but would follow the same principles of encapsulation and abstraction.
