import argparse
import os
from keystoneauth1.loading import base
def _register_plugin_argparse_arguments(parser, plugin):
    for opt in plugin.get_options():
        parser.add_argument(*opt.argparse_args, default=opt.argparse_default, metavar=opt.metavar, help=opt.help, dest='os_%s' % opt.dest)