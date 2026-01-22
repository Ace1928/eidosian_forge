import argparse
import code
import gzip
import ssl
import sys
import threading
import time
import zlib
from urllib.parse import urlparse
import websocket
class NonInteractive(RawInput):

    def write(self, data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def read(self) -> str:
        return self.raw_input('')