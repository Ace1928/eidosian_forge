import os
import sys
import ftplib
import warnings
from .utils import parse_url
@api_response.setter
def api_response(self, response):
    """Update the cached API response"""
    self._api_response = response