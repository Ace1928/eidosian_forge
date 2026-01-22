import base64
from email.message import EmailMessage
from typing import List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
def _prepare_draft_message(self, message: str, to: List[str], subject: str, cc: Optional[List[str]]=None, bcc: Optional[List[str]]=None) -> dict:
    draft_message = EmailMessage()
    draft_message.set_content(message)
    draft_message['To'] = ', '.join(to)
    draft_message['Subject'] = subject
    if cc is not None:
        draft_message['Cc'] = ', '.join(cc)
    if bcc is not None:
        draft_message['Bcc'] = ', '.join(bcc)
    encoded_message = base64.urlsafe_b64encode(draft_message.as_bytes()).decode()
    return {'message': {'raw': encoded_message}}