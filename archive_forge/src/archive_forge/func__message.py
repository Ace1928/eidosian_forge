import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def _message(text_content, replies):
    payload = {'text': text_content[:MAX_TEXT_CHARS]}
    if replies:
        payload['quick_replies'] = replies
    return payload