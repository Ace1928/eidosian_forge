import json
import re
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from queue import PriorityQueue, Queue
from threading import Thread
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, cast
from urllib.parse import unquote, urlparse, urlunparse
from docutils import nodes
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, TooManyRedirects
from sphinx.application import Sphinx
from sphinx.builders.dummy import DummyBuilder
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line
def write_linkstat(self, data: dict) -> None:
    self.json_outfile.write(json.dumps(data))
    self.json_outfile.write('\n')