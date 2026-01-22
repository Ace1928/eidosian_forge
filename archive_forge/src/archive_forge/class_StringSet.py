import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class StringSet(Branch):

    def __init__(self, info, name, case_flags=NOCASE):
        self.info = info
        self.name = name
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self._key = (self.__class__, self.name, self.case_flags)
        self.set_key = (name, self.case_flags)
        if self.set_key not in info.named_lists_used:
            info.named_lists_used[self.set_key] = len(info.named_lists_used)
        index = self.info.named_lists_used[self.set_key]
        items = self.info.kwargs[self.name]
        case_flags = self.case_flags
        encoding = self.info.flags & _ALL_ENCODINGS
        fold_flags = encoding | case_flags
        choices = []
        for string in items:
            if isinstance(string, str):
                string = [ord(c) for c in string]
            choices.append([Character(c, case_flags=case_flags) for c in string])
        choices.sort(key=len, reverse=True)
        self.branches = [Sequence(choice) for choice in choices]

    def dump(self, indent, reverse):
        print('{}STRING_SET {}{}'.format(INDENT * indent, self.name, CASE_TEXT[self.case_flags]))

    def __del__(self):
        self.info = None