from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadEvent(BaseModel):
    action: WebhookEvent_T
    scope: str