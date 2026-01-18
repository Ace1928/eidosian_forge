import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
@property
def tex_fn(self):
    fn = basename(self.source_fn).rsplit('.', 1)[0] + '.tex'
    return join(self.dest_dir, fn)