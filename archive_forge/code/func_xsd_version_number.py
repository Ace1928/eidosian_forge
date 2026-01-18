import sys
import os
import argparse
import logging
import pathlib
from urllib.error import URLError
import xmlschema
from xmlschema import XMLSchema, XMLSchema11, iter_errors, to_json, from_json, etree_tostring
from xmlschema.exceptions import XMLSchemaValueError
def xsd_version_number(value):
    if value not in ('1.0', '1.1'):
        raise argparse.ArgumentTypeError('%r is not a valid XSD version' % value)
    return value