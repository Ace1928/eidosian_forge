import sys
import traceback
from mako import compat
from mako import util
class RichTraceback:
    """Pull the current exception from the ``sys`` traceback and extracts
    Mako-specific template information.

    See the usage examples in :ref:`handling_exceptions`.

    """

    def __init__(self, error=None, traceback=None):
        self.source, self.lineno = ('', 0)
        if error is None or traceback is None:
            t, value, tback = sys.exc_info()
        if error is None:
            error = value or t
        if traceback is None:
            traceback = tback
        self.error = error
        self.records = self._init(traceback)
        if isinstance(self.error, (CompileException, SyntaxException)):
            self.source = self.error.source
            self.lineno = self.error.lineno
            self._has_source = True
        self._init_message()

    @property
    def errorname(self):
        return compat.exception_name(self.error)

    def _init_message(self):
        """Find a unicode representation of self.error"""
        try:
            self.message = str(self.error)
        except UnicodeError:
            try:
                self.message = str(self.error)
            except UnicodeEncodeError:
                self.message = self.error.args[0]
        if not isinstance(self.message, str):
            self.message = str(self.message, 'ascii', 'replace')

    def _get_reformatted_records(self, records):
        for rec in records:
            if rec[6] is not None:
                yield (rec[4], rec[5], rec[2], rec[6])
            else:
                yield tuple(rec[0:4])

    @property
    def traceback(self):
        """Return a list of 4-tuple traceback records (i.e. normal python
        format) with template-corresponding lines remapped to the originating
        template.

        """
        return list(self._get_reformatted_records(self.records))

    @property
    def reverse_records(self):
        return reversed(self.records)

    @property
    def reverse_traceback(self):
        """Return the same data as traceback, except in reverse order."""
        return list(self._get_reformatted_records(self.reverse_records))

    def _init(self, trcback):
        """format a traceback from sys.exc_info() into 7-item tuples,
        containing the regular four traceback tuple items, plus the original
        template filename, the line number adjusted relative to the template
        source, and code line from that line number of the template."""
        import mako.template
        mods = {}
        rawrecords = traceback.extract_tb(trcback)
        new_trcback = []
        for filename, lineno, function, line in rawrecords:
            if not line:
                line = ''
            try:
                line_map, template_lines, template_filename = mods[filename]
            except KeyError:
                try:
                    info = mako.template._get_module_info(filename)
                    module_source = info.code
                    template_source = info.source
                    template_filename = info.template_filename or info.template_uri or filename
                except KeyError:
                    new_trcback.append((filename, lineno, function, line, None, None, None, None))
                    continue
                template_ln = 1
                mtm = mako.template.ModuleInfo
                source_map = mtm.get_module_source_metadata(module_source, full_line_map=True)
                line_map = source_map['full_line_map']
                template_lines = [line_ for line_ in template_source.split('\n')]
                mods[filename] = (line_map, template_lines, template_filename)
            template_ln = line_map[lineno - 1]
            if template_ln <= len(template_lines):
                template_line = template_lines[template_ln - 1]
            else:
                template_line = None
            new_trcback.append((filename, lineno, function, line, template_filename, template_ln, template_line, template_source))
        if not self.source:
            for l in range(len(new_trcback) - 1, 0, -1):
                if new_trcback[l][5]:
                    self.source = new_trcback[l][7]
                    self.lineno = new_trcback[l][5]
                    break
            else:
                if new_trcback:
                    try:
                        with open(new_trcback[-1][0], 'rb') as fp:
                            encoding = util.parse_encoding(fp)
                            if not encoding:
                                encoding = 'utf-8'
                            fp.seek(0)
                            self.source = fp.read()
                        if encoding:
                            self.source = self.source.decode(encoding)
                    except IOError:
                        self.source = ''
                    self.lineno = new_trcback[-1][1]
        return new_trcback