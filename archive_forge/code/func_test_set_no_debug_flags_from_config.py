from .. import config, debug, tests
def test_set_no_debug_flags_from_config(self):
    self.assertDebugFlags([], b'')