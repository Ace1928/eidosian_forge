from argparse import ArgumentParser
from os.path import dirname
from os.path import isfile
import sys
from mako import exceptions
from mako.lookup import TemplateLookup
from mako.template import Template
def varsplit(var):
    if '=' not in var:
        return (var, '')
    return var.split('=', 1)