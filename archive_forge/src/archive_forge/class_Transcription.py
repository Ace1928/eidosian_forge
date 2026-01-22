from ..._models import BaseModel
class Transcription(BaseModel):
    text: str
    'The transcribed text.'