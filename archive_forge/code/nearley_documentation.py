import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
Converts Nearley grammars to Lark