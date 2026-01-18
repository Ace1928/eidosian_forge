import json
import logging
import os
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.listeners import printing
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.utils import misc
@property
def url_handle(self):
    if self._url_handle is None:
        self._url_handle = UrlCaller()
    return self._url_handle