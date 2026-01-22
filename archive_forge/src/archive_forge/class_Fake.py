import asyncio
import sqlite3
from pathlib import Path
from sqlite3 import OperationalError
from threading import Thread
from unittest import IsolatedAsyncioTestCase as TestCase, SkipTest
import aiosqlite
from .helpers import setup_logger
class Fake:

    def __str__(self):
        return str(TEST_DB)