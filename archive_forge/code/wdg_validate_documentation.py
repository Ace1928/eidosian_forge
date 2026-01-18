from io import StringIO
import subprocess
from paste.response import header_value
import re
import cgi

    Wraps the application in the WDG validator from
    http://www.htmlhelp.com/tools/validator/

    Validation errors are appended to the text of each page.
    You can configure this by giving the path to the validate
    executable (by default picked up from $PATH)
    