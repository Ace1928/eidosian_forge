from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field
from requests import Response
from requests.exceptions import HTTPError
class MaritalkHTTPError(HTTPError):

    def __init__(self, request_obj: Response) -> None:
        self.request_obj = request_obj
        try:
            response_json = request_obj.json()
            if 'detail' in response_json:
                api_message = response_json['detail']
            elif 'message' in response_json:
                api_message = response_json['message']
            else:
                api_message = response_json
        except Exception:
            api_message = request_obj.text
        self.message = api_message
        self.status_code = request_obj.status_code

    def __str__(self) -> str:
        status_code_meaning = HTTPStatus(self.status_code).phrase
        formatted_message = f'HTTP Error: {self.status_code} - {status_code_meaning}'
        formatted_message += f'\nDetail: {self.message}'
        return formatted_message