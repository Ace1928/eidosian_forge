import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_arg_features(self, arg_name, req_features):
    if not isinstance(req_features, str):
        self.dist_fatal("expected a string in '%s'" % arg_name)
    final_features = set()
    tokens = list(filter(None, re.split(self._parse_regex_arg, req_features)))
    append = True
    for tok in tokens:
        if tok[0] in ('#', '$'):
            self.dist_fatal(arg_name, "target groups and policies aren't allowed from arguments, only from dispatch-able sources")
        if tok == '+':
            append = True
            continue
        if tok == '-':
            append = False
            continue
        TOK = tok.upper()
        features_to = set()
        if TOK == 'NONE':
            pass
        elif TOK == 'NATIVE':
            native = self.cc_flags['native']
            if not native:
                self.dist_fatal(arg_name, "native option isn't supported by the compiler")
            features_to = self.feature_names(force_flags=native, macros=[('DETECT_FEATURES', 1)])
        elif TOK == 'MAX':
            features_to = self.feature_supported.keys()
        elif TOK == 'MIN':
            features_to = self.feature_min
        elif TOK in self.feature_supported:
            features_to.add(TOK)
        elif not self.feature_is_exist(TOK):
            self.dist_fatal(arg_name, ", '%s' isn't a known feature or option" % tok)
        if append:
            final_features = final_features.union(features_to)
        else:
            final_features = final_features.difference(features_to)
        append = True
    return final_features