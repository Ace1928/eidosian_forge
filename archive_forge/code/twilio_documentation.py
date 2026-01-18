from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Run body through Twilio and respond with message sid.

        Args:
            body: The text of the message you want to send. Can be up to 1,600
                characters in length.
            to: The destination phone number in
                [E.164](https://www.twilio.com/docs/glossary/what-e164) format for
                SMS/MMS or
                [Channel user address](https://www.twilio.com/docs/sms/channels#channel-addresses)
                for other 3rd-party channels.
        