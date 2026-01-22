from __future__ import annotations
import dataclasses
from typing import Any, BinaryIO, Dict, Optional, TYPE_CHECKING, Union
import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from requests.structures import CaseInsensitiveDict
from requests_toolbelt.multipart.encoder import MultipartEncoder  # type: ignore
from . import protocol
class RequestsBackend(protocol.Backend):

    def __init__(self, session: Optional[requests.Session]=None) -> None:
        self._client: requests.Session = session or requests.Session()

    @property
    def client(self) -> requests.Session:
        return self._client

    @staticmethod
    def prepare_send_data(files: Optional[Dict[str, Any]]=None, post_data: Optional[Union[Dict[str, Any], bytes, BinaryIO]]=None, raw: bool=False) -> SendData:
        if files:
            if post_data is None:
                post_data = {}
            else:
                if TYPE_CHECKING:
                    assert isinstance(post_data, dict)
                for k, v in post_data.items():
                    if isinstance(v, bool):
                        v = int(v)
                    if isinstance(v, (complex, float, int)):
                        post_data[k] = str(v)
            post_data['file'] = files.get('file')
            post_data['avatar'] = files.get('avatar')
            data = MultipartEncoder(fields=post_data)
            return SendData(data=data, content_type=data.content_type)
        if raw and post_data:
            return SendData(data=post_data, content_type='application/octet-stream')
        if TYPE_CHECKING:
            assert not isinstance(post_data, BinaryIO)
        return SendData(json=post_data, content_type='application/json')

    def http_request(self, method: str, url: str, json: Optional[Union[Dict[str, Any], bytes]]=None, data: Optional[Union[Dict[str, Any], MultipartEncoder]]=None, params: Optional[Any]=None, timeout: Optional[float]=None, verify: Optional[Union[bool, str]]=True, stream: Optional[bool]=False, **kwargs: Any) -> RequestsResponse:
        """Make HTTP request

        Args:
            method: The HTTP method to call ('get', 'post', 'put', 'delete', etc.)
            url: The full URL
            data: The data to send to the server in the body of the request
            json: Data to send in the body in json by default
            timeout: The timeout, in seconds, for the request
            verify: Whether SSL certificates should be validated. If
                the value is a string, it is the path to a CA file used for
                certificate validation.
            stream: Whether the data should be streamed

        Returns:
            A requests Response object.
        """
        response: requests.Response = self._client.request(method=method, url=url, params=params, data=data, timeout=timeout, stream=stream, verify=verify, json=json, **kwargs)
        return RequestsResponse(response=response)