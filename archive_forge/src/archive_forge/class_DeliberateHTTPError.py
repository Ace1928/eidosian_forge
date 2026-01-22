import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
class DeliberateHTTPError(Exception):

    def __init__(self, message, status_code: int=500):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code

    def get_response(self):
        return flask.Response(self.message, status=self.status_code)

    def __repr__(self):
        return f'DeliberateHTTPError({self.message!r}, {self.status_code!r})'