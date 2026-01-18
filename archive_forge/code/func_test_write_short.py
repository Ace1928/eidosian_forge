import unittest
import pygame
def test_write_short(self):
    if not self.midi_output:
        self.skipTest('No midi device')
    out = self.midi_output
    out.write_short(192)
    out.write_short(144, 65, 100)
    out.write_short(128, 65, 100)
    out.write_short(144)