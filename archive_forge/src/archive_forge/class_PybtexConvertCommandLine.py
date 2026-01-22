from __future__ import unicode_literals
from pybtex.cmdline import CommandLine, make_option, standard_option
class PybtexConvertCommandLine(CommandLine):
    prog = 'pybtex-convert'
    args = '[options] in_filename out_filename'
    description = 'convert between bibliography database formats'
    long_description = '\n\npybtex-convert converts bibliography database files between supported formats\n(currently BibTeX, BibTeXML and YAML).\n\n    '.strip()
    num_args = 2
    options = ((None, (standard_option('strict'), make_option('-f', '--from', dest='from_format', help='input format (%plugin_choices)', metavar='FORMAT', type='load_plugin', plugin_group='pybtex.database.input'), make_option('-t', '--to', dest='to_format', help='output format (%plugin_choices)', metavar='FORMAT', type='load_plugin', plugin_group='pybtex.database.output'), standard_option('keyless_entries'), make_option('--preserve-case', dest='preserve_case', action='store_true', help='do not convert identifiers to lower case'))), ('Encoding options', (standard_option('encoding'), standard_option('input_encoding'), standard_option('output_encoding'))))
    option_defaults = {'keyless_entries': False, 'preserve_case': False}

    def run(self, from_filename, to_filename, encoding, input_encoding, output_encoding, keyless_entries, **options):
        from pybtex.database.convert import convert
        convert(from_filename, to_filename, input_encoding=input_encoding or encoding, output_encoding=output_encoding or encoding, parser_options={'keyless_entries': keyless_entries}, **options)