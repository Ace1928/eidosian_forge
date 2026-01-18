import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher

        Starts the parsing process which detects records and notifies
        registered subscribers as it finds each IAB record.
        