from typing import Dict, Tuple, Union
from twisted.words.protocols.jabber.xmpp_stringprep import (
class InvalidFormat(Exception):
    """
    The given string could not be parsed into a valid Jabber Identifier (JID).
    """