import configobj
from . import bedding, cmdline, errors, globbing, osutils
class _IniBasedRulesSearcher(_RulesSearcher):

    def __init__(self, inifile):
        """Construct a _RulesSearcher based on an ini file.

        The content will be decoded as utf-8.

        :param inifile: the name of the file or a sequence of lines.
        """
        self._cfg = configobj.ConfigObj(inifile, encoding='utf-8')
        sections = self._cfg.keys()
        patterns = []
        self.pattern_to_section = {}
        for s in sections:
            if s.startswith(FILE_PREFS_PREFIX):
                file_patterns = cmdline.split(s[FILE_PREFS_PREFIX_LEN:])
                patterns.extend(file_patterns)
                for fp in file_patterns:
                    self.pattern_to_section[fp] = s
        if len(patterns) < len(sections):
            unknowns = [s for s in sections if not s.startswith(FILE_PREFS_PREFIX)]
            raise UnknownRules(unknowns)
        elif patterns:
            self._globster = globbing._OrderedGlobster(patterns)
        else:
            self._globster = None

    def get_items(self, path):
        """See _RulesSearcher.get_items."""
        if self._globster is None:
            return ()
        pat = self._globster.match(path)
        if pat is None:
            return ()
        else:
            all = self._cfg[self.pattern_to_section[pat]]
            return tuple(all.items())

    def get_selected_items(self, path, names):
        """See _RulesSearcher.get_selected_items."""
        if self._globster is None:
            return ()
        pat = self._globster.match(path)
        if pat is None:
            return ()
        else:
            all = self._cfg[self.pattern_to_section[pat]]
            return tuple(((k, all.get(k)) for k in names))