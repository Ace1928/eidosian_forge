import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def _exception_data():
    """
    Return exception information:

    - the exception's class name;
    - the exception object;
    - the name of the file containing the offending code;
    - the line number of the offending code;
    - the function name of the offending code.
    """
    type, value, traceback = sys.exc_info()
    while traceback.tb_next:
        traceback = traceback.tb_next
    code = traceback.tb_frame.f_code
    return (type.__name__, value, code.co_filename, traceback.tb_lineno, code.co_name)