import warnings
from io import StringIO
import time
from urllib.parse import urlencode
from urllib.request import build_opener, install_opener
from urllib.request import urlopen
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler
from urllib.request import Request
from Bio import BiopythonWarning
from Bio._utils import function_with_previous
Extract a tuple of RID, RTOE from the 'please wait' page (PRIVATE).

    The NCBI FAQ pages use TOE for 'Time of Execution', so RTOE is probably
    'Request Time of Execution' and RID would be 'Request Identifier'.
    