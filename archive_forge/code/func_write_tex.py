import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
def write_tex(self, data, fmt, image_size, texture_size, mipmap=False, formatinfo=None):
    infos = {'datalen': len(data), 'image_size': image_size, 'texture_size': texture_size, 'mipmap': mipmap, 'format': fmt}
    if formatinfo:
        infos['formatinfo'] = formatinfo
    header = json.dumps(infos, indent=0, separators=(',', ':'))
    header = header.replace('\n', '')
    with open(self.tex_fn, 'wb') as fd:
        fd.write('KTEX')
        fd.write(pack('I', len(header)))
        fd.write(header)
        fd.write(data)
    print('Done! Compressed texture written at {}'.format(self.tex_fn))
    pprint(infos)