import json
import re
import typing as ty
from requests import exceptions as _rex
class ConflictException(HttpException):
    """HTTP 409 Conflict."""