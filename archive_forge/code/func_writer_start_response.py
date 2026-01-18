from io import StringIO
import subprocess
from paste.response import header_value
import re
import cgi
def writer_start_response(status, headers, exc_info=None):
    response.extend((status, headers))
    start_response(status, headers, exc_info)
    return output.write