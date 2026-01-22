import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
class HfHubHTTPError(HTTPError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in HF Hub.

    Any HTTPError is converted at least into a `HfHubHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request id from "X-Request-Id" header if exists.
    - Server error message from the header "X-Error-Message".
    - Server error message if we can found one in the response body.

    Example:
    ```py
        import requests
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("
`create_commit` expects the repository to exist.")
            raise
    ```
    """
    request_id: Optional[str] = None
    server_message: Optional[str] = None

    def __init__(self, message: str, response: Optional[Response]=None):
        if response is not None:
            self.request_id = response.headers.get('X-Request-Id')
            try:
                server_data = response.json()
            except JSONDecodeError:
                server_data = {}
            server_message_from_headers = response.headers.get('X-Error-Message')
            server_message_from_body = server_data.get('error')
            server_multiple_messages_from_body = '\n'.join((error['message'] for error in server_data.get('errors', []) if 'message' in error))
            _server_message = ''
            if server_message_from_headers is not None:
                _server_message += server_message_from_headers + '\n'
            if server_message_from_body is not None:
                if isinstance(server_message_from_body, list):
                    server_message_from_body = '\n'.join(server_message_from_body)
                if server_message_from_body not in _server_message:
                    _server_message += server_message_from_body + '\n'
            if server_multiple_messages_from_body is not None:
                if server_multiple_messages_from_body not in _server_message:
                    _server_message += server_multiple_messages_from_body + '\n'
            _server_message = _server_message.strip()
            if _server_message != '':
                self.server_message = _server_message
        super().__init__(_format_error_message(message, request_id=self.request_id, server_message=self.server_message), response=response, request=response.request if response is not None else None)

    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]