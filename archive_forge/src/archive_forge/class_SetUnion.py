import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class SetUnion(SetBase):
    _opcode = {(NOCASE, False): OP.SET_UNION, (IGNORECASE, False): OP.SET_UNION_IGN, (FULLCASE, False): OP.SET_UNION, (FULLIGNORECASE, False): OP.SET_UNION_IGN, (NOCASE, True): OP.SET_UNION_REV, (IGNORECASE, True): OP.SET_UNION_IGN_REV, (FULLCASE, True): OP.SET_UNION_REV, (FULLIGNORECASE, True): OP.SET_UNION_IGN_REV}
    _op_name = 'SET_UNION'

    def optimise(self, info, reverse, in_set=False):
        items = []
        for m in self.items:
            m = m.optimise(info, reverse, in_set=True)
            if isinstance(m, SetUnion) and m.positive:
                items.extend(m.items)
            else:
                items.append(m)
        if len(items) == 1:
            i = items[0]
            return i.with_flags(positive=i.positive == self.positive, case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
        self.items = tuple(items)
        return self._handle_case_folding(info, in_set)

    def _compile(self, reverse, fuzzy):
        flags = 0
        if self.positive:
            flags |= POSITIVE_OP
        if self.zerowidth:
            flags |= ZEROWIDTH_OP
        if fuzzy:
            flags |= FUZZY_OP
        characters, others = (defaultdict(list), [])
        for m in self.items:
            if isinstance(m, Character):
                characters[m.positive].append(m.value)
            else:
                others.append(m)
        code = [(self._opcode[self.case_flags, reverse], flags)]
        for positive, values in characters.items():
            flags = 0
            if positive:
                flags |= POSITIVE_OP
            if len(values) == 1:
                code.append((OP.CHARACTER, flags, values[0]))
            else:
                code.append((OP.STRING, flags, len(values)) + tuple(values))
        for m in others:
            code.extend(m.compile())
        code.append((OP.END,))
        return code

    def matches(self, ch):
        m = any((i.matches(ch) for i in self.items))
        return m == self.positive