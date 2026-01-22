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
class TestImageInsertion(BaseDatabaseTestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        with open('/home/lloyd/EVIE/scripts/image_interconversion_gui/test_image.png', 'rb') as image_file:
            image_data = image_file.read()
        await insert_compressed_image('hash123', 'png', image_data)

    async def test_insert_compressed_image_success(self):
        with open('/home/lloyd/EVIE/scripts/image_interconversion_gui/test_image.png', 'rb') as image_file:
            image_data = image_file.read()
        await insert_compressed_image('hash123', 'png', image_data)

    async def test_insert_compressed_image_failure(self):
        with self.assertRaises(ValueError):
            await insert_compressed_image('hash123', 'jpg', b'')