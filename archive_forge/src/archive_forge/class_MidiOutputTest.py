import unittest
import pygame
class MidiOutputTest(unittest.TestCase):
    __tags__ = ['interactive']

    def setUp(self):
        import pygame.midi
        pygame.midi.init()
        m_out_id = pygame.midi.get_default_output_id()
        if m_out_id != -1:
            self.midi_output = pygame.midi.Output(m_out_id)
        else:
            self.midi_output = None

    def tearDown(self):
        if self.midi_output:
            self.midi_output.close()
        pygame.midi.quit()

    def test_Output(self):
        i = pygame.midi.get_default_output_id()
        if self.midi_output:
            self.assertEqual(self.midi_output.device_id, i)
        i = pygame.midi.get_default_input_id()
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, -1)
        self.assertRaises(TypeError, pygame.midi.Output, '1234')
        self.assertRaises(OverflowError, pygame.midi.Output, pow(2, 99))

    def test_note_off(self):
        if self.midi_output:
            out = self.midi_output
            out.note_on(5, 30, 0)
            out.note_off(5, 30, 0)
            with self.assertRaises(ValueError) as cm:
                out.note_off(5, 30, 25)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
            with self.assertRaises(ValueError) as cm:
                out.note_off(5, 30, -1)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')

    def test_note_on(self):
        if self.midi_output:
            out = self.midi_output
            out.note_on(5, 30, 0)
            out.note_on(5, 42, 10)
            with self.assertRaises(ValueError) as cm:
                out.note_on(5, 30, 25)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
            with self.assertRaises(ValueError) as cm:
                out.note_on(5, 30, -1)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')

    def test_set_instrument(self):
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.set_instrument(5)
        out.set_instrument(42, channel=2)
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(-6)
        self.assertEqual(str(cm.exception), 'Undefined instrument id: -6')
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(156)
        self.assertEqual(str(cm.exception), 'Undefined instrument id: 156')
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(5, -1)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(5, 16)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')

    def test_write(self):
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.write([[[192, 0, 0], 20000]])
        out.write([[[192], 20000]])
        out.write([[[192, 0, 0], 20000], [[144, 60, 100], 20500]])
        out.write([])
        verrry_long = [[[144, 60, i % 100], 20000 + 100 * i] for i in range(1024)]
        out.write(verrry_long)
        too_long = [[[144, 60, i % 100], 20000 + 100 * i] for i in range(1025)]
        self.assertRaises(IndexError, out.write, too_long)
        with self.assertRaises(TypeError) as cm:
            out.write('Non sens ?')
        error_msg = "unsupported operand type(s) for &: 'str' and 'int'"
        self.assertEqual(str(cm.exception), error_msg)
        with self.assertRaises(TypeError) as cm:
            out.write(["Hey what's that?"])
        self.assertEqual(str(cm.exception), error_msg)

    def test_write_short(self):
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.write_short(192)
        out.write_short(144, 65, 100)
        out.write_short(128, 65, 100)
        out.write_short(144)

    def test_write_sys_ex(self):
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.write_sys_ex(pygame.midi.time(), [240, 125, 16, 17, 18, 19, 247])

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

    def test_close(self):
        if not self.midi_output:
            self.skipTest('No midi device')
        self.assertIsNotNone(self.midi_output._output)
        self.midi_output.close()
        self.assertIsNone(self.midi_output._output)

    def test_abort(self):
        if not self.midi_output:
            self.skipTest('No midi device')
        self.assertEqual(self.midi_output._aborted, 0)
        self.midi_output.abort()
        self.assertEqual(self.midi_output._aborted, 1)