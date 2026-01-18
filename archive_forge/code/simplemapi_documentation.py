import os
from ctypes import *
Post an e-mail message using Simple MAPI

    recipient - string: address to send to (multiple addresses separated with a semicolon)
    subject   - string: subject header
    body      - string: message text
    attach    - string: files to attach (multiple attachments separated with a semicolon)
    