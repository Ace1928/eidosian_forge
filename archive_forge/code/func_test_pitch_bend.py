import unittest
import pygame
def test_pitch_bend(self):
    if not self.midi_output:
        self.skipTest('No midi device')
    out = self.midi_output
    with self.assertRaises(ValueError) as cm:
        out.pitch_bend(5, channel=-1)
    self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
    with self.assertRaises(ValueError) as cm:
        out.pitch_bend(5, channel=16)
    with self.assertRaises(ValueError) as cm:
        out.pitch_bend(-10001, 1)
    self.assertEqual(str(cm.exception), 'Pitch bend value must be between -8192 and +8191, not -10001.')
    with self.assertRaises(ValueError) as cm:
        out.pitch_bend(10665, 2)