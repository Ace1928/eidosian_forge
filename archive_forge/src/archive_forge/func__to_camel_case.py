import abc
import re
def _to_camel_case(self, name):
    return self.regex.sub(lambda m: m.group(1).upper(), name)