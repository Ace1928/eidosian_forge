from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _placeholders(connection, names):
    if connection is None:
        debug('connection is None, default to using qmark paramstyle')
        placeholders = ', '.join(['?'] * len(names))
    else:
        mod = __import__(connection.__class__.__module__)
        if not hasattr(mod, 'paramstyle'):
            debug('module %r from connection %r has no attribute paramstyle, defaulting to qmark', mod, connection)
            placeholders = ', '.join(['?'] * len(names))
        elif mod.paramstyle == 'qmark':
            debug('found paramstyle qmark')
            placeholders = ', '.join(['?'] * len(names))
        elif mod.paramstyle in ('format', 'pyformat'):
            debug('found paramstyle pyformat')
            placeholders = ', '.join(['%s'] * len(names))
        elif mod.paramstyle == 'numeric':
            debug('found paramstyle numeric')
            placeholders = ', '.join([':' + str(i + 1) for i in range(len(names))])
        elif mod.paramstyle == 'named':
            debug('found paramstyle named')
            placeholders = ', '.join([':%s' % name for name in names])
        else:
            debug('found unexpected paramstyle %r, defaulting to qmark', mod.paramstyle)
            placeholders = ', '.join(['?'] * len(names))
    return placeholders