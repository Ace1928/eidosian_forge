import asyncio
import enum
import logging
from io import BytesIO
import aiodns
import aiodns.error
import aiohttp
from . import defaults
from .utils import parse_mta_sts_record, parse_mta_sts_policy, is_plaintext, filter_text
from .constants import HARD_RESP_LIMIT, CHUNK
class BadSTSPolicy(Exception):
    pass