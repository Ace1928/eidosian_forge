from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayload(BaseModel):
    event: WebhookPayloadEvent
    repo: WebhookPayloadRepo
    discussion: Optional[WebhookPayloadDiscussion] = None
    comment: Optional[WebhookPayloadComment] = None
    webhook: WebhookPayloadWebhook
    movedTo: Optional[WebhookPayloadMovedTo] = None