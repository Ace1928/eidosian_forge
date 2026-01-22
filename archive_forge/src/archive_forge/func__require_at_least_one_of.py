import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def _require_at_least_one_of(self, *params):
    argument_present = False
    for param in params:
        if hasattr(self, param):
            if getattr(self, param):
                argument_present = True
    if argument_present is False:
        raise ArgumentsRequired(*params)