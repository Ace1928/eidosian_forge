import logging
from datetime import datetime as dt
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.slack.base import SlackBaseTool
from langchain_community.tools.slack.utils import UTC_FORMAT
class ScheduleMessageSchema(BaseModel):
    """Input for ScheduleMessageTool."""
    message: str = Field(..., description='The message to be sent.')
    channel: str = Field(..., description='The channel, private group, or IM channel to send message to.')
    timestamp: str = Field(..., description='The datetime for when the message should be sent in the  following format: YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date  and time components, and the time zone offset is specified as ±hh:mm.  For example: "2023-06-09T10:30:00+03:00" represents June 9th,  2023, at 10:30 AM in a time zone with a positive offset of 3  hours from Coordinated Universal Time (UTC).')