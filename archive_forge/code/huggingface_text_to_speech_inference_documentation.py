import logging
import os
import uuid
from datetime import datetime
from typing import Callable, Literal, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.tools import BaseTool
Function to generate unique file names.