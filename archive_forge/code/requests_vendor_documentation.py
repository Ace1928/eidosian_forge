from io import BytesIO
from requests import Session
from ..client import (
from ..errors import GitProtocolError, NotGitRepository
Requests HTTP client support for Dulwich.

To use this implementation as the HTTP implementation in Dulwich, override
the dulwich.client.HttpGitClient attribute:

  >>> from dulwich import client as _mod_client
  >>> from dulwich.contrib.requests_vendor import RequestsHttpGitClient
  >>> _mod_client.HttpGitClient = RequestsHttpGitClient

This implementation is experimental and does not have any tests.
