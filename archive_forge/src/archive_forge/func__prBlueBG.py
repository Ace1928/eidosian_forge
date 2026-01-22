import json
import uuid
import websocket
import time
import threading
from parlai.core.params import ParlaiParser
def _prBlueBG(text):
    """
    Print given in text with a blue background.

    :param text: The text to be printed
    """
    print('\x1b[44m{}\x1b[0m'.format(text), sep='')