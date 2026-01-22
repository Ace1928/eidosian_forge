from __future__ import absolute_import
import types
from . import Errors
class Alt(RE):
    """Alt(re1, re2, re3...) is an RE which matches either |re1| or
    |re2| or |re3|..."""

    def __init__(self, *re_list):
        self.re_list = re_list
        nullable = 0
        match_nl = 0
        nullable_res = []
        non_nullable_res = []
        i = 1
        for re in re_list:
            self.check_re(i, re)
            if re.nullable:
                nullable_res.append(re)
                nullable = 1
            else:
                non_nullable_res.append(re)
            if re.match_nl:
                match_nl = 1
            i += 1
        self.nullable_res = nullable_res
        self.non_nullable_res = non_nullable_res
        self.nullable = nullable
        self.match_nl = match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        for re in self.nullable_res:
            re.build_machine(m, initial_state, final_state, match_bol, nocase)
        if self.non_nullable_res:
            if match_bol:
                initial_state = self.build_opt(m, initial_state, BOL)
            for re in self.non_nullable_res:
                re.build_machine(m, initial_state, final_state, 0, nocase)

    def calc_str(self):
        return 'Alt(%s)' % ','.join(map(str, self.re_list))