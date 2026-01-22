import sys
from threading import Thread
from .version import version as current
from packaging.version import parse as parse_version
import ssl
from urllib import request
class Phoner(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.answer = None

    def run(self):
        this_version = parse_version(current)
        latest = None
        try:
            with request.urlopen(version_url) as response:
                latest = response.read().decode('ascii').strip()
                latest_version = parse_version(latest)
        except Exception:
            return
        if latest and latest_version > this_version:
            self.answer = (latest, current)