import csv
import json
import logging
class JsonDecoder(object):
    """A decoder for JSON formatted data."""

    def decode(self, x):
        return json.loads(x)