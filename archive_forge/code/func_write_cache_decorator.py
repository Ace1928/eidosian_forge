import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_cache_decorator(self, node_or_pagetag, name, args, buffered, identifiers, inline=False, toplevel=False):
    """write a post-function decorator to replace a rendering
        callable with a cached version of itself."""
    self.printer.writeline('__M_%s = %s' % (name, name))
    cachekey = node_or_pagetag.parsed_attributes.get('cache_key', repr(name))
    cache_args = {}
    if self.compiler.pagetag is not None:
        cache_args.update(((pa[6:], self.compiler.pagetag.parsed_attributes[pa]) for pa in self.compiler.pagetag.parsed_attributes if pa.startswith('cache_') and pa != 'cache_key'))
    cache_args.update(((pa[6:], node_or_pagetag.parsed_attributes[pa]) for pa in node_or_pagetag.parsed_attributes if pa.startswith('cache_') and pa != 'cache_key'))
    if 'timeout' in cache_args:
        cache_args['timeout'] = int(eval(cache_args['timeout']))
    self.printer.writeline('def %s(%s):' % (name, ','.join(args)))
    pass_args = ['%s=%s' % ((a.split('=')[0],) * 2) if '=' in a else a for a in args]
    self.write_variable_declares(identifiers, toplevel=toplevel, limit=node_or_pagetag.undeclared_identifiers())
    if buffered:
        s = "context.get('local').cache._ctx_get_or_create(%s, lambda:__M_%s(%s),  context, %s__M_defname=%r)" % (cachekey, name, ','.join(pass_args), ''.join(['%s=%s, ' % (k, v) for k, v in cache_args.items()]), name)
        s = self.create_filter_callable(self.compiler.buffer_filters, s, False)
        self.printer.writelines('return ' + s, None)
    else:
        self.printer.writelines("__M_writer(context.get('local').cache._ctx_get_or_create(%s, lambda:__M_%s(%s), context, %s__M_defname=%r))" % (cachekey, name, ','.join(pass_args), ''.join(['%s=%s, ' % (k, v) for k, v in cache_args.items()]), name), "return ''", None)