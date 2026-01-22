import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _result_msg_type(result: 'pb.Result') -> str:
    msg_type = str(result.WhichOneof('result_type'))
    if msg_type == 'response':
        response = result.response
        msg_type = str(response.WhichOneof('response_type'))
    return msg_type