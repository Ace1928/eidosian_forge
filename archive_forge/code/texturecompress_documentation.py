import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink

Texture compression tool
========================

This tool is designed to compress images into:

- PVRTC (PowerVR Texture Compression), mostly iOS devices
- ETC1 (Ericson compression), working on all GLES2/Android devices

Usage
-----

In order to compress a texture::

    texturecompress.py [--dir <directory>] <format> <image.png>

This will create a `image.tex` file with a json header that contains all the
image information and the compressed data.

TODO
----

Support more format, such as:

- S3TC (already supported in Kivy)
- DXT1 (already supported in Kivy)
