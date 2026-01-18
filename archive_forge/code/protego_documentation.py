import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
Replaces a %xx escape with equivalent binary sequence.