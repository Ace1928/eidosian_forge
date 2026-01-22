import datetime
import json
import random
import re
import time
import traceback
import uuid
import typing as t
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
Quotes a value for PowerShell.

    Quotes a value to be safely used by a PowerShell expression. The input
    string because something that is safely wrapped in single quotes.

    Args:
        s: The string to quote.

    Returns:
        (text_type): The quoted string value.
    