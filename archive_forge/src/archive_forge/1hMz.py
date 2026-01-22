"""
sound_fetch.py

Author: Lloyd Handyside
Creation Date: 2024-03-31

Description:
This module is designed to download soundfont files from a specified URL. It handles the fetching of soundfont JSON configurations, downloading instrument JSON files, and the associated pitch/velocity MP3 files for each instrument. It ensures that only missing files are downloaded to optimize bandwidth usage.

"""

import json
import os
import urllib.request
import logging
import functools
import time
import hashlib
from typing import Callable, List


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


setup_logging()
logging.debug("Starting soundfont download process")

__all__ = ["log_execution_time", "get_pitches_array", "download_file", "file_md5"]


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log the execution time of a function.

    Parameters:
    - func (Callable): The function to be decorated.

    Returns:
    - Callable: The wrapped function with execution time logging.

    This decorator calculates the execution time of the function it decorates and logs it.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result

    return wrapper


def get_pitches_array(min_pitch: int, max_pitch: int) -> List[int]:
    """
    Generates a list of pitches within the specified range.

    Parameters:
    - min_pitch (int): The minimum pitch value.
    - max_pitch (int): The maximum pitch value.

    Returns:
    - List[int]: A list of integers representing pitches.

    Raises:
    - ValueError: If min_pitch is greater than max_pitch.

    Example:
    >>> get_pitches_array(60, 62)
    [60, 61, 62]
    """
    if min_pitch > max_pitch:
        raise ValueError("min_pitch cannot be greater than max_pitch")
    return list(range(min_pitch, max_pitch + 1))


def file_md5(file_path: str) -> str:
    """Compute the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, dest_path: str, expected_md5: str = None):
    """Download a file if it doesn't exist or if its checksum doesn't match."""
    if os.path.exists(dest_path):
        if expected_md5 is None or file_md5(dest_path) == expected_md5:
            logging.info(f"File already exists and is verified: {dest_path}")
            return
        else:
            logging.info(
                f"File exists but doesn't match the expected checksum, re-downloading: {dest_path}"
            )

    try:
        with urllib.request.urlopen(url) as response:
            file_data = response.read()
            # Verify checksum after download if expected_md5 is provided
            if expected_md5:
                actual_md5 = hashlib.md5(file_data).hexdigest()
                if actual_md5 != expected_md5:
                    logging.error(f"Checksum mismatch for downloaded file: {dest_path}")
                    return
            with open(dest_path, "wb") as out_file:
                out_file.write(file_data)
        logging.info(f"Downloaded and verified file from {url} to {dest_path}")
    except urllib.error.URLError as e:
        logging.error(f"Network error when trying to download {url}: {e}")
    except Exception as e:
        logging.error(f"Failed to download {url} due to {e}")


# Constants for base URL and soundfont path
base_url = "https://storage.googleapis.com/magentadata/js/soundfonts"
soundfont_path = "sgm_plus"
soundfont_json_url = f"{base_url}/{soundfont_path}/soundfont.json"
logging.debug(f"Attempting to download soundfont.json from {soundfont_json_url}")

# Attempt to download soundfont.json if it does not exist locally
try:
    if not os.path.exists("soundfont.json"):
        download_file(soundfont_json_url, "soundfont.json")
    else:
        with open("soundfont.json", "rb") as file:  # Ensure reading in binary mode
            soundfont_json = file.read()
except Exception as e:
    logging.critical(f"Unable to proceed without soundfont.json due to {e}")

# Parse soundfont.json
soundfont_data = (
    None  # Initialize soundfont_data to None to handle potential unbound variable issue
)
try:
    # Check if 'soundfont_json' is defined to avoid "possibly unbound" error
    if "soundfont_json" in locals():
        # Decoding the binary data to a string before parsing it as JSON
        soundfont_data_str = soundfont_json.decode("utf-8")
        soundfont_data = json.loads(soundfont_data_str)
    else:
        logging.error("soundfont_json is not defined, cannot parse soundfont.json")
except json.JSONDecodeError as e:
    logging.error(f"Failed to parse soundfont.json due to {e}")
    # Ensuring soundfont_data remains None if an exception occurs

if soundfont_data is not None:
    for instrument_id, instrument_name in soundfont_data["instruments"].items():
        logging.debug(f"Processing instrument: {instrument_name}")
        if not os.path.isdir(instrument_name):
            os.makedirs(instrument_name)
        instrument_json: bytes = b""  # Corrected type annotation
        instrument_path = f"{soundfont_path}/{instrument_name}"
        try:
            if not os.path.exists(f"{instrument_name}/instrument.json"):
                instrument_json_url = f"{base_url}/{instrument_path}/instrument.json"
                download_file(instrument_json_url, f"{instrument_name}/instrument.json")
            else:
                with open(f"{instrument_name}/instrument.json", "rb") as file:
                    instrument_json = file.read()
        except Exception as e:
            logging.error(
                f"Failed to download or read {instrument_name}/instrument.json due to {e}"
            )
        try:
            instrument_data = json.loads(instrument_json)
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse {instrument_name}/instrument.json due to {e}"
            )
            instrument_data = None
        if instrument_data is not None:
            for velocity in instrument_data["velocities"]:
                pitches = get_pitches_array(
                    instrument_data["minPitch"], instrument_data["maxPitch"]
                )
                for pitch in pitches:
                    file_name = f"p{pitch}_v{velocity}.mp3"
                    if not os.path.exists(f"{instrument_name}/{file_name}"):
                        file_url = f"{base_url}/{instrument_path}/{file_name}"
                        download_file(file_url, f"{instrument_name}/{file_name}")
                        logging.info(f"Downloaded {instrument_name}/{file_name}")
else:
    logging.error("Failed to parse soundfont.json")

"""
TODO:
- Refactor the script to encapsulate logic within functions or classes for better reusability and clarity.
- Implement a logging mechanism to replace print statements for more granular debugging and operational insight.
- Enhance error handling by specifying exception types and adding more comprehensive error messages.
- Consider adding progress indicators for file downloads to improve user experience.

Known Issues:
- Global variables are used extensively, which could lead to issues when integrating with other modules or scaling the script.
- Lack of a main guard (`if __name__ == "__main__":`) makes the script execute all operations on import, which might not be desirable.
"""
if __name__ == "__main__":
    setup_logging()
    # Main logic here
    try:
        if not os.path.exists("soundfont.json"):
            download_file(soundfont_json_url, "soundfont.json")
        else:
            with open("soundfont.json", "rb") as file:
                soundfont_json = file.read()
        if "soundfont_json" in locals():
            soundfont_data_str = soundfont_json.decode("utf-8")
            soundfont_data = json.loads(soundfont_data_str)
        if soundfont_data is not None:
            for instrument_id, instrument_name in soundfont_data["instruments"].items():
                if not os.path.isdir(instrument_name):
                    os.makedirs(instrument_name)
                instrument_json_url = (
                    f"{base_url}/{soundfont_path}/{instrument_name}/instrument.json"
                )
                download_file(instrument_json_url, f"{instrument_name}/instrument.json")
                with open(f"{instrument_name}/instrument.json", "rb") as file:
                    instrument_json = file.read()
                instrument_data = json.loads(instrument_json)
                if instrument_data is not None:
                    for velocity in instrument_data["velocities"]:
                        pitches = get_pitches_array(
                            instrument_data["minPitch"], instrument_data["maxPitch"]
                        )
                        for pitch in pitches:
                            file_name = f"p{pitch}_v{velocity}.mp3"
                            file_url = f"{base_url}/{soundfont_path}/{instrument_name}/{file_name}"
                            download_file(file_url, f"{instrument_name}/{file_name}")
    except Exception as e:
        logging.critical(f"Critical failure: {e}")
