from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def apply_overrides(spectree, pathkeystr, value):
    """Apply command specification overrides to a previously constructed
     parse function tree."""
    if '.' not in pathkeystr:
        raise UserError('Invalid override key {}'.format(pathkeystr))
    pathstr, keystr = pathkeystr.rsplit('.', 1)
    pathparts = pathstr.split('/')
    while pathparts:
        if not isinstance(spectree, CommandSpec):
            raise InternalError('Invalid spectree entry of type {}'.format(type(spectree)))
        pathkey = pathparts.pop(0)
        if pathkey.startswith('parg'):
            if pathparts:
                raise UserError('Invalid override key {} contains path components after pspec'.format(pathkeystr))
            try:
                idx = int(pathkey[len('parg'):].strip('[]'))
            except ValueError:
                raise UserError('Invalid override key {} contains non-int pspec'.format(pathkeystr))
            if idx > len(spectree.pargs):
                raise UserError('Invalid override key {} is out of bounds'.format(pathkeystr))
            repl = {keystr: value}
            spectree.pargs[idx] = spectree.pargs[idx].replace(**repl)
            return
        if pathkey not in spectree.kwargs:
            raise UserError('Invalid override key {} at {}'.format(pathkeystr, pathkey))
        spectree = spectree.kwargs[pathkey]
    if not hasattr(spectree, keystr):
        raise UserError('Invalid override key {}, keypart {} is not valid for type {}'.format(pathkeystr, keystr, type(spectree)))
    try:
        setattr(spectree, keystr, value)
    except AttributeError:
        raise UserError("Invalid override key {}, can't set attribute {} for type {}".format(pathkeystr, keystr, type(spectree)))