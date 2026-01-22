from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadDiscussion(ObjectId):
    num: int
    author: ObjectId
    url: WebhookPayloadUrl
    title: str
    isPullRequest: bool
    status: DiscussionStatus_T
    changes: Optional[WebhookPayloadDiscussionChanges] = None
    pinned: Optional[bool] = None