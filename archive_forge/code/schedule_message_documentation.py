import logging
from datetime import datetime as dt
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.slack.base import SlackBaseTool
from langchain_community.tools.slack.utils import UTC_FORMAT
Tool for scheduling a message in Slack.