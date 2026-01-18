import asyncio
import aiosqlite
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
from typing import Optional, List, Tuple, AsyncContextManager, NoReturn, AsyncGenerator
import backoff
from contextlib import asynccontextmanager
import io
from PIL import Image  # Import PIL.Image for image handling
import unittest
from unittest import IsolatedAsyncioTestCase
from core_services import ConfigManager, LoggingManager, EncryptionManager
from image_processing import (
def validate_pagination_params(offset: int, limit: int) -> NoReturn:
    """
    Validates the pagination parameters for database queries.

    Ensures that the offset is non-negative and the limit is positive, raising appropriate exceptions if not.

    Parameters:
        offset (int): The offset from where to start fetching the records.
        limit (int): The maximum number of records to fetch.

    Raises:
        TypeError: If either offset or limit is not an integer.
        ValueError: If offset is negative or limit is not positive.
    """
    if not isinstance(offset, int) or not isinstance(limit, int):
        raise TypeError('Offset and limit must be integers.')
    if offset < 0 or limit <= 0:
        raise ValueError('Offset must be non-negative and limit must be positive.')