import logging
import logging.config
import logging.handlers
import sys
import time
import asyncio
import aiofiles
from typing import (
import pathlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
import functools
from functools import wraps
import tracemalloc
import inspect
from inspect import signature, Parameter
from IndegoValidation import AsyncValidationException, ValidationRules

                A synchronous wrapper function that directly executes the decorated synchronous function without event loop manipulation.
                This function retains core functionalities such as validation, retrying with exponential backoff, performance logging, and dynamic retry strategy adjustment,
                while simplifying the execution path for synchronous functions.

                Args:
                    *args: Positional arguments for the decorated function.
                    **kwargs: Keyword arguments for the decorated function.

                Returns:
                    Any: The result of the decorated function execution.
                