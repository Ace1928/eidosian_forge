from functools import lru_cache, partial
import re
from pyparsing import (
from matplotlib import _api
@lru_cache
def _make_fontconfig_parser():

    def comma_separated(elem):
        return elem + ZeroOrMore(Suppress(',') + elem)
    family = Regex(f'([^{_family_punc}]|(\\\\[{_family_punc}]))*')
    size = Regex('([0-9]+\\.?[0-9]*|\\.[0-9]+)')
    name = Regex('[a-z]+')
    value = Regex(f'([^{_value_punc}]|(\\\\[{_value_punc}]))*')
    prop = Group(name + Suppress('=') + comma_separated(value) | name)
    return Optional(comma_separated(family)('families')) + Optional('-' + comma_separated(size)('sizes')) + ZeroOrMore(':' + prop('properties*')) + StringEnd()