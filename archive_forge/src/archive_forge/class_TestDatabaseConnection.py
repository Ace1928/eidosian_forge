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
class TestDatabaseConnection(BaseDatabaseTestCase):

    async def test_get_db_connection_failure(self):
        original_db_path = DatabaseConfig.DB_PATH
        DatabaseConfig.DB_PATH = 'invalid/path/to/database.db'
        with self.assertRaises(Exception):
            async with get_db_connection() as _:
                pass
        DatabaseConfig.DB_PATH = original_db_path