"""
Logging Utilities for RWKV-Runner Backend.

This module provides logging functionalities for the RWKV-Runner backend application,
including functions for quick logging of requests and responses, and middleware for
logging HTTP requests. It utilizes the standard logging library and custom JSON encoding
for Pydantic models and Enums.

Author: Your Name
Date: YYYY-MM-DD
"""

from dotenv import load_dotenv

load_dotenv()

import json
import logging
import logging.handlers
from typing import Any
from fastapi import Request
from pydantic import BaseModel
from enum import Enum

# Configure the logger for the application
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s\n%(message)s")
fh = logging.handlers.RotatingFileHandler(
    "api.log", mode="a", maxBytes=3 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
fh.setFormatter(formatter)
logger.addHandler(fh)


class ClsEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder for logging purposes.

    This encoder is specifically designed to handle serialization of Pydantic models
    and Enums, facilitating their logging by converting them into their dictionary
    representation or value, respectively.

    Attributes:
        obj (Any): The object to be serialized.

    Returns:
        Any: The serialized object.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def quick_log(request: Request, body: Any, response: str) -> None:
    """
    Logs basic information about a request and its response.

    Parameters:
        request (Request): The FastAPI request object.
        body (Any): The request body, potentially a Pydantic model.
        response (str): The response data as a string.

    This function logs the client's IP, the request URL, the request body (if any),
    and the response. It's designed for quick and simple logging of API interactions.
    """
    try:
        logger.info(
            f"Client: {request.client if request else ''}\nUrl: {request.url if request else ''}\n"
            + (
                f"Body: {json.dumps(body.__dict__, ensure_ascii=False, cls=ClsEncoder)}\n"
                if body
                else ""
            )
            + (f"Data:\n{response}\n" if response else "")
        )
    except Exception as e:
        logger.error(f"Error in quick_log: {e}")


async def log_middleware(request: Request) -> None:
    """
    Asynchronous middleware for logging HTTP requests.

    Parameters:
        request (Request): The FastAPI request object.

    This middleware logs the client's IP, the request URL, and the request body.
    It's intended to be used with FastAPI applications for detailed request logging.
    """
    try:
        body = await request.body()
        logger.info(
            f"Client: {request.client}\nUrl: {request.url}\nBody: {body.decode('utf-8')}\n"
        )
    except Exception as e:
        logger.error(f"Error in log_middleware: {e}")


# TODO:
# - Implement more granular logging levels based on environment variables.
# - Add support for structured logging (e.g., JSON format) for better integration with log management tools.
# - Consider adding context managers for logging to encapsulate request-response cycles.
