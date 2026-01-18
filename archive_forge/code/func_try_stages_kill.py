import unittest
from bpython.curtsiesfrontend.manual_readline import (
def try_stages_kill(self, strings, func):
    if not all(('|' in s for s in strings)):
        raise ValueError("Need to use '|' to specify cursor")
    stages = [(s.index('|'), s.replace('|', '')) for s in strings]
    for (initial_pos, initial), (final_pos, final) in zip(stages[:-1], stages[1:]):
        self.assertEqual(func(initial_pos, initial)[:-1], (final_pos, final))