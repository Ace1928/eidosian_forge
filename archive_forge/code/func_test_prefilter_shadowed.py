import pytest
from IPython.core.prefilter import AutocallChecker
def test_prefilter_shadowed():

    def dummy_magic(line):
        pass
    prev_automagic_state = ip.automagic
    ip.automagic = True
    ip.autocall = 0
    try:
        for name in ['if', 'zip', 'get_ipython']:
            ip.register_magic_function(dummy_magic, magic_name=name)
            res = ip.prefilter(name + ' foo')
            assert res == name + ' foo'
            del ip.magics_manager.magics['line'][name]
        for name in ['fi', 'piz', 'nohtypi_teg']:
            ip.register_magic_function(dummy_magic, magic_name=name)
            res = ip.prefilter(name + ' foo')
            assert res != name + ' foo'
            del ip.magics_manager.magics['line'][name]
    finally:
        ip.automagic = prev_automagic_state