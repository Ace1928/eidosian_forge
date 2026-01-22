import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
class PositionsDotConv(Dot2PGFConv):
    """A converter that returns a dictionary with node positions

    Returns a dictionary with node name as key and a (x, y) tuple as value.
    """

    def output(self):
        positions = {}
        for node in self.nodes:
            pos = getattr(node, 'pos', None)
            if pos:
                try:
                    positions[node.name] = [int(p) for p in pos.split(',')]
                except ValueError:
                    positions[node.name] = [float(p) for p in pos.split(',')]
        return positions