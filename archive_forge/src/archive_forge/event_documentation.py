import json
import argparse
from cliff import command
from datetime import datetime
from iso8601 import iso8601
from iso8601 import ParseError
Post an event to Vitrage