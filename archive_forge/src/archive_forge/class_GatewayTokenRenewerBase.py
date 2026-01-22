from __future__ import annotations
import asyncio
import json
import logging
import os
import typing as ty
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from socket import gaierror
from jupyter_events import EventLogger
from tornado import web
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPResponse
from traitlets import (
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
class GatewayTokenRenewerBase(ABC, LoggingConfigurable, metaclass=GatewayTokenRenewerMeta):
    """
    Abstract base class for refreshing tokens used between this server and a Gateway
    server.  Implementations requiring additional configuration can extend their class
    with appropriate configuration values or convey those values via appropriate
    environment variables relative to the implementation.
    """

    @abstractmethod
    def get_token(self, auth_header_key: str, auth_scheme: ty.Union[str, None], auth_token: str, **kwargs: ty.Any) -> str:
        """
        Given the current authorization header key, scheme, and token, this method returns
        a (potentially renewed) token for use against the Gateway server.
        """