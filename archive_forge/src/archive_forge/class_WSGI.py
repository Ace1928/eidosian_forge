from __future__ import print_function
import argparse
import functools
import sys
class WSGI(object):

    @classmethod
    def app(self):
        return functools.partial(application, data='Hello World')