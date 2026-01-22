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

    Initializes the database and migrates the schema to the latest version.

    This function ensures that the database schema is up to date by calling the migrate_schema function. It is intended to be run at application startup to prepare the database for use.

    Raises:
        Exception: If initializing the database fails.
    