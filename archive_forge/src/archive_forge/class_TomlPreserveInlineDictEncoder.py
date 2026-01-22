import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
class TomlPreserveInlineDictEncoder(TomlEncoder):

    def __init__(self, _dict=dict):
        super(TomlPreserveInlineDictEncoder, self).__init__(_dict, True)