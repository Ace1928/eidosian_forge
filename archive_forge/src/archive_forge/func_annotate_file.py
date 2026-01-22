from __future__ import unicode_literals
import argparse
import io
import logging
import os
import sys
import cmakelang
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang import render
def annotate_file(config, infile, outfile, outfmt=None):
    """
  Parse the input cmake file, re-format it, and print to the output file.
  """
    infile_content = infile.read()
    if config.format.line_ending == 'auto':
        detected = __main__.detect_line_endings(infile_content)
        config = config.clone()
        config.format.set_line_ending(detected)
    tokens = lex.tokenize(infile_content)
    parse_db = parse.funs.get_parse_db()
    parse_db.update(parse.funs.get_funtree(config.parse.fn_spec))
    ctx = parse.ParseContext(parse_db)
    parse_tree = parse.parse(tokens, ctx)
    if outfmt == 'page':
        html_content = render.get_html(parse_tree, fullpage=True)
        outfile.write(html_content)
        return
    if outfmt == 'stub':
        html_content = render.get_html(parse_tree, fullpage=False)
        outfile.write(html_content)
        return
    if outfmt == 'iframe':
        html_content = render.get_html(parse_tree, fullpage=True)
        wrap_lines = EMBED_TPL.split('\n')
        for line in wrap_lines[:2]:
            outfile.write(line)
            outfile.write('\n')
        outfile.write(html_content)
        for line in wrap_lines[3:]:
            outfile.write(line)
            outfile.write('\n')
        return
    raise ValueError('Invalid output format: {}'.format(outfmt))