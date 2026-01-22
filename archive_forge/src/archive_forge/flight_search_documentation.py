import logging
from datetime import datetime as dt
from typing import Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.amadeus.base import AmadeusBaseTool
Tool for searching for a single flight between two airports.