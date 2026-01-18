from __future__ import absolute_import, division, print_function
import argparse
import gzip
import pathlib
import shutil
import subprocess
import sys
from urllib import request
from xml.etree import ElementTree
import yaml
An argument parser that displays help on error.