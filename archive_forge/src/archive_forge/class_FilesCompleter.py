from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
from .compat import str, sys_encoding
class FilesCompleter(object):
    """
    File completer class, optionally takes a list of allowed extensions
    """

    def __init__(self, allowednames=(), directories=True):
        if isinstance(allowednames, (str, bytes)):
            allowednames = [allowednames]
        self.allowednames = [x.lstrip('*').lstrip('.') for x in allowednames]
        self.directories = directories

    def __call__(self, prefix, **kwargs):
        completion = []
        if self.allowednames:
            if self.directories:
                files = _call(['bash', '-c', "compgen -A directory -- '{p}'".format(p=prefix)])
                completion += [f + '/' for f in files]
            for x in self.allowednames:
                completion += _call(['bash', '-c', "compgen -A file -X '!*.{0}' -- '{p}'".format(x, p=prefix)])
        else:
            completion += _call(['bash', '-c', "compgen -A file -- '{p}'".format(p=prefix)])
            anticomp = _call(['bash', '-c', "compgen -A directory -- '{p}'".format(p=prefix)])
            completion = list(set(completion) - set(anticomp))
            if self.directories:
                completion += [f + '/' for f in anticomp]
        return completion