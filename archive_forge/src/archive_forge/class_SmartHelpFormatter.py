import argparse as orig_argparse
import warnings
from autopage import argparse
class SmartHelpFormatter(argparse.HelpFormatter):
    """Smart help formatter to output raw help message if help contain 
.

    Some command help messages maybe have multiple line content, the built-in
    argparse.HelpFormatter wrap and split the content according to width, and
    ignore 
 in the raw help message, it merge multiple line content in one
    line to output, that looks messy. SmartHelpFormatter keep the raw help
    message format if it contain 
, and wrap long line like HelpFormatter
    behavior.
    """

    def _split_lines(self, text, width):
        lines = text.splitlines() if '\n' in text else [text]
        wrap_lines = []
        for each_line in lines:
            wrap_lines.extend(super(SmartHelpFormatter, self)._split_lines(each_line, width))
        return wrap_lines