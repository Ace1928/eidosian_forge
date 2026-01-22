import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
class ArgumentsRequired(ArgumentRequired):

    def __init__(self, *params):
        self.params = params

    def __str__(self):
        returnstring = 'Specify at least one of these arguments: '
        for param in self.params:
            returnstring = returnstring + '"--%s" ' % param
        return returnstring