from datetime import datetime as dt
from typing import List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT
class O365SendEvent(O365BaseTool):
    """Tool for sending calendar events in Office 365."""
    name: str = 'send_event'
    description: str = 'Use this tool to create and send an event with the provided event fields.'
    args_schema: Type[SendEventSchema] = SendEventSchema

    def _run(self, body: str, attendees: List[str], subject: str, start_datetime: str, end_datetime: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        schedule = self.account.schedule()
        calendar = schedule.get_default_calendar()
        event = calendar.new_event()
        event.body = body
        event.subject = subject
        event.start = dt.strptime(start_datetime, UTC_FORMAT)
        event.end = dt.strptime(end_datetime, UTC_FORMAT)
        for attendee in attendees:
            event.attendees.add(attendee)
        event.save()
        output = 'Event sent: ' + str(event)
        return output