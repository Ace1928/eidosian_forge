import json
import os
import logging
from typing import List, Dict, Optional, Any


# Setup for comprehensive logging throughout the application.
def setup_logger() -> None:
    """
    Configures the logging system, creating a log file to record all significant application events, errors, and operations.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="md_to_jsonl_conversion.log",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def parse_md_content(file_content: str) -> List[Dict[str, str]]:
    """
    Extracts conversation pairs from Markdown content, structuring them into a list of dictionaries for further processing.

    Args:
        file_content (str): The string content of a Markdown file.

    Returns:
        A list of dictionaries, each representing a conversation pair with 'input' and 'output' keys.
    """
    logger = logging.getLogger("parse_md_content")
    conversations = []
    user_input, assistant_response = "", ""
    capture_mode: Optional[str] = None

    for line in file_content.split("\n"):
        if line.startswith("## USER"):
            if assistant_response:  # Ensures previous conversation pairs are captured
                conversations.append(
                    {"input": user_input.strip(), "output": assistant_response.strip()}
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

    logger.info(f"Extracted {len(conversations)} conversations from Markdown content.")
    return conversations


def convert_conversations_to_jsonl(
    conversations: List[Dict[str, str]], output_path: str
) -> None:
    """
    Serializes conversation pairs to a JSON Lines file, ensuring each pair is properly formatted and saved.

    Args:
        conversations: A list of dictionaries, where each dictionary contains 'input' and 'output' of a conversation.
        output_path: File path for the output JSONL file.
    """
    logger = logging.getLogger("convert_to_jsonl")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")
        logger.info(f"Conversations successfully written to {output_path}")
    except Exception as e:
        logger.error(
            f"Failed to write conversations to JSONL due to {e}", exc_info=True
        )


def process_md_file(md_path: str, output_dir: str) -> None:
    """
    Converts a single Markdown file into JSON Lines format, saving the result in a specified directory.

    Args:
        md_path: Path to the Markdown file.
        output_dir: Directory to save the output JSONL file.
    """
    logger = logging.getLogger("process_md_file")
    output_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(md_path))[0] + ".jsonl"
    )

    try:
        with open(md_path, "r", encoding="utf-8") as md_file:
            file_content = md_file.read()
            conversations = parse_md_content(file_content)
            convert_conversations_to_jsonl(conversations, output_path)
    except Exception as e:
        logger.error(f"Failed to process {md_path} due to {e}", exc_info=True)


def process_md_directory(input_dir: str, output_dir: str) -> None:
    """
    Recursively processes Markdown files within a directory, converting them to JSON Lines format for model fine-tuning.

    Args:
        input_dir: Directory containing the Markdown files.
        output_dir: Target directory for the output JSONL files.
    """
    logger = logging.getLogger("process_md_directory")
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                process_md_file(md_path, output_dir)

    logger.info(f"All Markdown files in {input_dir} have been processed.")


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger("main")
    input_directory = "path/to/md_files_directory"  # Adjust this path as needed.
    output_directory = "path/to/output_jsonl_directory"  # Adjust this path as needed.

    try:
        process_md_directory(input_directory, output_directory)
        logger.info("Markdown conversion process completed successfully.")
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during the Markdown conversion process.",
            exc_info=True,
        )
