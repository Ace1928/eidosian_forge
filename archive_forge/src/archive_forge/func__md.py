import sys
import argparse
from .renderers.rst import RSTRenderer
from .renderers.markdown import MarkdownRenderer
from . import (
def _md(args):
    if args.plugin:
        plugins = args.plugin
    else:
        plugins = ['strikethrough', 'footnotes', 'table', 'speedup']
    if args.renderer == 'rst':
        renderer = RSTRenderer()
    elif args.renderer == 'markdown':
        renderer = MarkdownRenderer()
    else:
        renderer = args.renderer
    return create_markdown(escape=args.escape, hard_wrap=args.hardwrap, renderer=renderer, plugins=plugins)