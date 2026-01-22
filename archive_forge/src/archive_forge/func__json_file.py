from __future__ import absolute_import
import argparse
import json
import sys
from jsonschema._reflect import namedAny
from jsonschema.validators import validator_for
def _json_file(path):
    with open(path) as file:
        return json.load(file)