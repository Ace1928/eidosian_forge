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
class BaseDatabaseTestCase(IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        await initialize_test_database()

    async def asyncTearDown(self):
        async with aiosqlite.connect(DatabaseConfig.DB_PATH) as db:
            await db.execute('DROP TABLE IF EXISTS images')
            await db.commit()