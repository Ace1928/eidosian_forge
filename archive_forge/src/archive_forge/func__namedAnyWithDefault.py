from __future__ import absolute_import
import argparse
import json
import sys
from jsonschema._reflect import namedAny
from jsonschema.validators import validator_for
def _namedAnyWithDefault(name):
    if '.' not in name:
        name = 'jsonschema.' + name
    return namedAny(name)