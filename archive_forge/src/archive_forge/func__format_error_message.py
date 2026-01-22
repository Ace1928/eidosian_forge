import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
def _format_error_message(message: str, request_id: Optional[str], server_message: Optional[str]) -> str:
    """
    Format the `HfHubHTTPError` error message based on initial message and information
    returned by the server.

    Used when initializing `HfHubHTTPError`.
    """
    if server_message is not None and len(server_message) > 0 and (server_message.lower() not in message.lower()):
        if '\n\n' in message:
            message += '\n' + server_message
        else:
            message += '\n\n' + server_message
    if request_id is not None and str(request_id).lower() not in message.lower():
        request_id_message = f' (Request ID: {request_id})'
        if '\n' in message:
            newline_index = message.index('\n')
            message = message[:newline_index] + request_id_message + message[newline_index:]
        else:
            message += request_id_message
    return message