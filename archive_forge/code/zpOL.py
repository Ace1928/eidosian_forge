"""
sound_fetch.py

Author: Lloyd Handyside
Creation Date: 2024-03-31

Description:
This module is designed to download soundfont files from a specified URL. It handles the fetching of soundfont JSON configurations, downloading instrument JSON files, and the associated pitch/velocity MP3 files for each instrument. It ensures that only missing files are downloaded to optimize bandwidth usage.

Functions:
- setup_logging: Configures the logging format and level.
- log_execution_time: Decorator for logging the execution time of functions.
- get_pitches_array: Generates a list of pitches within a specified range.
- file_md5: Asynchronously computes the MD5 hash of a file.
- download_file: Asynchronously downloads a file with optional MD5 verification.
- main: The main asynchronous entry point of the script.

Constants:
- base_url: The base URL for soundfont files.
- soundfont_path: The path to the soundfont files.
- soundfont_json_url: The full URL to the soundfont JSON configuration.

Known Issues and TODOs are documented at the end of the module.
"""

import aiohttp
import aiofiles
import asyncio
import json
import os
import logging
import functools
import time
import hashlib
from typing import Callable, List


def setup_logging():
    """
    Configures the logging format and level for the application.

    This function sets up the logging configuration to use a specific format and info level, ensuring that all log messages throughout the application are consistent and informative.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


setup_logging()
logging.debug("Starting soundfont download process")

__all__ = [
    "setup_logging",
    "log_execution_time",
    "get_pitches_array",
    "download_file",
    "file_md5",
    "main",
]


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log the execution time of a function.

    Parameters:
    - func (Callable): The function to be decorated.

    Returns:
    - Callable: The wrapped function with execution time logging.

    This decorator calculates the execution time of the function it decorates and logs it, providing insights into performance.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
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


async def file_md5(file_path: str) -> str:
    """
    Asynchronously compute the MD5 hash of a file.

    Parameters:
    - file_path (str): The path to the file whose MD5 hash is to be computed.

    Returns:
    - str: The computed MD5 hash in hexadecimal format.
    """
    hash_md5 = hashlib.md5()
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            chunk = await f.read(4096)
            if not chunk:
                break
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


async def download_file(
    session: aiohttp.ClientSession, url: str, dest_path: str, expected_md5: str = None
) -> None:
    """
    Asynchronously download a file if it doesn't exist or if its checksum doesn't match.

    Parameters:
    - session (aiohttp.ClientSession): The session used for HTTP requests.
    - url (str): The URL of the file to download.
    - dest_path (str): The destination path for the downloaded file.
    - expected_md5 (str, optional): The expected MD5 checksum of the file.

    This function checks if the file already exists and matches the expected MD5 checksum before downloading to save bandwidth.
    """
    if os.path.exists(dest_path):
        if expected_md5 is None or await file_md5(dest_path) == expected_md5:
            logging.info(f"File already exists and is verified: {dest_path}")
            return
        else:
            logging.info(
                f"File exists but doesn't match the expected checksum, re-downloading: {dest_path}"
            )

    try:
        async with session.get(url) as response:
            file_data = await response.read()
            if expected_md5:
                actual_md5 = hashlib.md5(file_data).hexdigest()
                if actual_md5 != expected_md5:
                    logging.error(f"Checksum mismatch for downloaded file: {dest_path}")
                    return
            async with aiofiles.open(dest_path, "wb") as out_file:
                await out_file.write(file_data)
        logging.info(f"Downloaded and verified file from {url} to {dest_path}")
    except aiohttp.ClientError as e:
        logging.error(f"Network error when trying to download {url}: {e}")
    except Exception as e:
        logging.error(f"Failed to download {url} due to {e}")


# Constants for base URL and soundfont path
base_url = "https://storage.googleapis.com/magentadata/js/soundfonts"
soundfont_path = "sgm_plus"
soundfont_json_url = f"{base_url}/{soundfont_path}/soundfont.json"
logging.debug(f"Attempting to download soundfont.json from {soundfont_json_url}")


async def main() -> None:
    """
    The main asynchronous entry point of the script.

    This function orchestrates the download process, including fetching the soundfont JSON configuration and downloading necessary files.
    """
    setup_logging()
    async with aiohttp.ClientSession() as session:
        # Attempt to download soundfont.json if it does not exist locally
        try:
            if not os.path.exists("assets/sound-font/soundfont.json"):
                await download_file(
                    session, soundfont_json_url, "assets/sound-font/soundfont.json"
                )
            else:
                async with aiofiles.open(
                    "assets/sound-font/soundfont.json", "rb"
                ) as file:
                    soundfont_json = await file.read()
        except Exception as e:
            logging.critical(f"Unable to proceed without soundfont.json due to {e}")

        # Parse soundfont.json
        soundfont_data = None
        try:
            if soundfont_json:
                soundfont_data_str = soundfont_json.decode("utf-8")
                soundfont_data = json.loads(soundfont_data_str)
            else:
                logging.error(
                    "soundfont_json is not defined, cannot parse soundfont.json"
                )
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse soundfont.json due to {e}")

        if soundfont_data is not None:
            for instrument_id, instrument_name in soundfont_data["instruments"].items():
                logging.debug(f"Processing instrument: {instrument_name}")
                os.makedirs(f"assets/sound-font/{instrument_name}", exist_ok=True)
                instrument_json: bytes = b""
                instrument_path = f"{soundfont_path}/{instrument_name}"
                try:
                    if not os.path.exists(
                        f"assets/sound-font/{instrument_name}/instrument.json"
                    ):
                        instrument_json_url = (
                            f"{base_url}/{instrument_path}/instrument.json"
                        )
                        await download_file(
                            session,
                            instrument_json_url,
                            f"assets/sound-font/{instrument_name}/instrument.json",
                        )
                    else:
                        async with aiofiles.open(
                            f"assets/sound-font/{instrument_name}/instrument.json", "rb"
                        ) as file:
                            instrument_json = await file.read()
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
                            file_url = f"{base_url}/{instrument_path}/{file_name}"
                            await download_file(
                                session,
                                file_url,
                                f"assets/sound-font/{instrument_name}/{file_name}",
                            )
                            logging.info(f"Downloaded {instrument_name}/{file_name}")
        else:
            logging.error("Failed to parse soundfont.json")


"""
TODO:
- Implement progress indicators for file downloads to improve user experience.
- Refactor global variables to use a more encapsulated approach, reducing potential side-effects.
- Enhance the script's modularity by organizing related functionalities into classes or separate modules where applicable.
- Introduce a configuration file or command-line arguments to allow dynamic setting of URLs and paths, enhancing flexibility.
- Expand the error handling to include more specific exception types for finer-grained control and response to errors.
- Consider implementing a caching mechanism to avoid re-downloading files that have not changed, further optimizing bandwidth usage.

Known Issues:
- The script executes all operations on import due to the lack of a main guard, which might not be desirable in all contexts.
- Extensive use of global variables could lead to issues when integrating with other modules or scaling the script.
- Error handling currently does not specify exception types, which could lead to less informative error messages and difficulty in debugging.
- The current logging setup is basic and might not provide enough granularity for debugging complex issues or understanding detailed execution flow.
"""

if __name__ == "__main__":
    asyncio.run(main())
