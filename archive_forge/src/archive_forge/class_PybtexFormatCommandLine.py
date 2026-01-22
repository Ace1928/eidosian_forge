from __future__ import unicode_literals
from pybtex.cmdline import CommandLine, standard_option
class PybtexFormatCommandLine(CommandLine):
    prog = 'pybtex-format'
    args = '[options] in_filename out_filename'
    description = 'format bibliography database as human-readable text'
    long_description = '\n\npybtex-format formats bibliography database as human-readable text.\n    '.strip()
    num_args = 2
    options = ((None, (standard_option('strict'), standard_option('bib_format'), standard_option('output_backend'), standard_option('min_crossrefs'), standard_option('keyless_entries'), standard_option('style'))), ('Pythonic style options', (standard_option('label_style'), standard_option('name_style'), standard_option('sorting_style'), standard_option('abbreviate_names'))), ('Encoding options', (standard_option('encoding'), standard_option('input_encoding'), standard_option('output_encoding'))))
    option_defaults = {'keyless_entries': False}

    def run(self, from_filename, to_filename, encoding, input_encoding, output_encoding, keyless_entries, **options):
        from pybtex.database.format import format_database
        format_database(from_filename, to_filename, input_encoding=input_encoding or encoding, output_encoding=output_encoding or encoding, parser_options={'keyless_entries': keyless_entries}, **options)