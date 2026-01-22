import os
from editorconfig import VERSION
from editorconfig.exceptions import PathError, VersionError
from editorconfig.ini import EditorConfigParser
class EditorConfigHandler(object):
    """
    Allows locating and parsing of EditorConfig files for given filename

    In addition to the constructor a single public method is provided,
    ``get_configurations`` which returns the EditorConfig options for
    the ``filepath`` specified to the constructor.

    """

    def __init__(self, filepath, conf_filename='.editorconfig', version=VERSION):
        """Create EditorConfigHandler for matching given filepath"""
        self.filepath = filepath
        self.conf_filename = conf_filename
        self.version = version
        self.options = None

    def get_configurations(self):
        """
        Find EditorConfig files and return all options matching filepath

        Special exceptions that may be raised by this function include:

        - ``VersionError``: self.version is invalid EditorConfig version
        - ``PathError``: self.filepath is not a valid absolute filepath
        - ``ParsingError``: improperly formatted EditorConfig file found

        """
        self.check_assertions()
        path, filename = os.path.split(self.filepath)
        conf_files = get_filenames(path, self.conf_filename)
        for filename in conf_files:
            parser = EditorConfigParser(self.filepath)
            parser.read(filename)
            old_options = self.options
            self.options = parser.options
            if old_options:
                self.options.update(old_options)
            if parser.root_file:
                break
        self.preprocess_values()
        return self.options

    def check_assertions(self):
        """Raise error if filepath or version have invalid values"""
        if not os.path.isabs(self.filepath):
            raise PathError('Input file must be a full path name.')
        if self.version is not None and self.version[:3] > VERSION[:3]:
            raise VersionError('Required version is greater than the current version.')

    def preprocess_values(self):
        """Preprocess option values for consumption by plugins"""
        opts = self.options
        for name in ['end_of_line', 'indent_style', 'indent_size', 'insert_final_newline', 'trim_trailing_whitespace', 'charset']:
            if name in opts:
                opts[name] = opts[name].lower()
        if opts.get('indent_style') == 'tab' and (not 'indent_size' in opts) and (self.version >= (0, 10, 0)):
            opts['indent_size'] = 'tab'
        if 'indent_size' in opts and 'tab_width' not in opts and (opts['indent_size'] != 'tab'):
            opts['tab_width'] = opts['indent_size']
        if 'indent_size' in opts and 'tab_width' in opts and (opts['indent_size'] == 'tab'):
            opts['indent_size'] = opts['tab_width']