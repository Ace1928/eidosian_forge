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
class TestExecuteDBQuery(BaseDatabaseTestCase):

    async def test_execute_db_query_success(self):
        query = 'CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)'
        await execute_db_query(query)

    async def test_execute_db_query_failure(self):
        query = 'INVALID SQL QUERY'
        with self.assertRaises(Exception):
            await execute_db_query(query)