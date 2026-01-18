import argparse
import os
import sys
from bokeh.__main__ import main as bokeh_entry_point
from bokeh.command.subcommands.serve import Serve as BkServe
from bokeh.command.util import die
from bokeh.util.strings import nice_join
from .. import __version__
from .bundle import Bundle
from .convert import Convert
from .oauth_secret import OAuthSecret
from .serve import Serve
def transform_cmds(argv):
    """
    Allows usage with anaconda-project by remapping the argv list provided
    into arguments accepted by Bokeh 0.12.7 or later.
    """
    replacements = {'--anaconda-project-host': '--allow-websocket-origin', '--anaconda-project-port': '--port', '--anaconda-project-address': '--address'}
    if 'PANEL_AE5_CDN' in os.environ:
        os.environ['BOKEH_RESOURCES'] = 'cdn'
    transformed = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in replacements.keys():
            transformed.append(replacements[arg])
        elif arg == '--anaconda-project-iframe-hosts':
            skip = True
            continue
        elif arg.startswith('--anaconda-project'):
            continue
        else:
            transformed.append(arg)
    return transformed