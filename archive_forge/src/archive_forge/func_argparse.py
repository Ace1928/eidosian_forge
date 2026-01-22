import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def argparse(self, name, args):
    if not name and 'name' in self.defaults:
        args['name'] = self.defaults['name']
    if type(name) is bytes or type(name) is str:
        args['name'] = name
    elif len(name) == 1:
        if name[0]:
            args['name'] = name[0]
    if defaults['server_rotate'] and type(defaults['server']) == types.ListType:
        defaults['server'] = defaults['server'][1:] + defaults['server'][:1]
    for i in list(defaults.keys()):
        if i not in args:
            if i in self.defaults:
                args[i] = self.defaults[i]
            else:
                args[i] = defaults[i]
    if type(args['server']) == bytes or type(args['server']) == str:
        args['server'] = [args['server']]
    self.args = args