from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def all_domains(self):
    resp = self.send('domains/')
    return resp['domains']