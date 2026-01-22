import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
class Etc1Tool(Tool):

    def __init__(self, options):
        super(Etc1Tool, self).__init__(options)
        self.etc1tool = None
        self.locate_etc1tool()

    def locate_etc1tool(self):
        search_directories = [environ.get('ANDROIDSDK', '/')]
        search_directories += environ.get('PATH', '').split(':')
        for directory in search_directories:
            fn = join(directory, 'etc1tool')
            if not exists(fn):
                fn = join(directory, 'tools', 'etc1tool')
                if not exists(fn):
                    continue
            print('Found texturetool at {}'.format(directory))
            self.etc1tool = fn
            return
        if self.etc1tool is None:
            print('Error: Unable to locate "etc1tool".\nMake sure that "etc1tool" is available in your PATH.\nOr export the path of your Android SDK to ANDROIDSDK')
            exit(1)

    def compress(self):
        image = Image.open(self.source_fn)
        w, h = image.size
        print('Image size is {}x{}'.format(*image.size))
        w2 = self.nearest_pow2(w)
        h2 = self.nearest_pow2(h)
        print('Nearest power-of-2 size is {}x{}'.format(w2, h2))
        raw_tex_fn = self.tex_fn + '.raw'
        cmd = [self.etc1tool, self.source_fn, '--encodeNoHeader', '-o', raw_tex_fn]
        try:
            self.runcmd(cmd)
            with open(raw_tex_fn, 'rb') as fd:
                data = fd.read()
        finally:
            if exists(raw_tex_fn):
                unlink(raw_tex_fn)
        self.write_tex(data, 'etc1_rgb8', (w, h), (w2, h2), self.options.mipmap)