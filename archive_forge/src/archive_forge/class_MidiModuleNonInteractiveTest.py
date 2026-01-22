import unittest
import pygame
class MidiModuleNonInteractiveTest(unittest.TestCase):
    """Midi module tests that do not require midi hardware or midi.init().

    See MidiModuleTest for interactive module tests.
    """

    def setUp(self):
        import pygame.midi

    def test_midiin(self):
        """Ensures the MIDIIN event id exists in the midi module.

        The MIDIIN event id can be accessed via the midi module for backward
        compatibility.
        """
        self.assertEqual(pygame.midi.MIDIIN, pygame.MIDIIN)
        self.assertEqual(pygame.midi.MIDIIN, pygame.locals.MIDIIN)
        self.assertNotEqual(pygame.midi.MIDIIN, pygame.MIDIOUT)
        self.assertNotEqual(pygame.midi.MIDIIN, pygame.locals.MIDIOUT)

    def test_midiout(self):
        """Ensures the MIDIOUT event id exists in the midi module.

        The MIDIOUT event id can be accessed via the midi module for backward
        compatibility.
        """
        self.assertEqual(pygame.midi.MIDIOUT, pygame.MIDIOUT)
        self.assertEqual(pygame.midi.MIDIOUT, pygame.locals.MIDIOUT)
        self.assertNotEqual(pygame.midi.MIDIOUT, pygame.MIDIIN)
        self.assertNotEqual(pygame.midi.MIDIOUT, pygame.locals.MIDIIN)

    def test_MidiException(self):
        """Ensures the MidiException is raised as expected."""

        def raiseit():
            raise pygame.midi.MidiException('Hello Midi param')
        with self.assertRaises(pygame.midi.MidiException) as cm:
            raiseit()
        self.assertEqual(cm.exception.parameter, 'Hello Midi param')

    def test_midis2events(self):
        """Ensures midi events are properly converted to pygame events."""
        MIDI_DATA = 0
        MD_STATUS = 0
        MD_DATA1 = 1
        MD_DATA2 = 2
        MD_DATA3 = 3
        TIMESTAMP = 1
        midi_events = (((192, 0, 1, 2), 20000), ((144, 60, 1000, 'string_data'), 20001), (('0', '1', '2', '3'), '4'))
        expected_num_events = len(midi_events)
        for device_id in range(3):
            pg_events = pygame.midi.midis2events(midi_events, device_id)
            self.assertEqual(len(pg_events), expected_num_events)
            for i, pg_event in enumerate(pg_events):
                midi_event = midi_events[i]
                midi_event_data = midi_event[MIDI_DATA]
                self.assertEqual(pg_event.__class__.__name__, 'Event')
                self.assertEqual(pg_event.type, pygame.MIDIIN)
                self.assertEqual(pg_event.status, midi_event_data[MD_STATUS])
                self.assertEqual(pg_event.data1, midi_event_data[MD_DATA1])
                self.assertEqual(pg_event.data2, midi_event_data[MD_DATA2])
                self.assertEqual(pg_event.data3, midi_event_data[MD_DATA3])
                self.assertEqual(pg_event.timestamp, midi_event[TIMESTAMP])
                self.assertEqual(pg_event.vice_id, device_id)

    def test_midis2events__missing_event_data(self):
        """Ensures midi events with missing values are handled properly."""
        midi_event_missing_data = ((192, 0, 1), 20000)
        midi_event_missing_timestamp = ((192, 0, 1, 2),)
        for midi_event in (midi_event_missing_data, midi_event_missing_timestamp):
            with self.assertRaises(ValueError):
                events = pygame.midi.midis2events([midi_event], 0)

    def test_midis2events__extra_event_data(self):
        """Ensures midi events with extra values are handled properly."""
        midi_event_extra_data = ((192, 0, 1, 2, 'extra'), 20000)
        midi_event_extra_timestamp = ((192, 0, 1, 2), 20000, 'extra')
        for midi_event in (midi_event_extra_data, midi_event_extra_timestamp):
            with self.assertRaises(ValueError):
                events = pygame.midi.midis2events([midi_event], 0)

    def test_midis2events__extra_event_data_missing_timestamp(self):
        """Ensures midi events with extra data and no timestamps are handled
        properly.
        """
        midi_event_extra_data_no_timestamp = ((192, 0, 1, 2, 'extra'),)
        with self.assertRaises(ValueError):
            events = pygame.midi.midis2events([midi_event_extra_data_no_timestamp], 0)

    def test_conversions(self):
        """of frequencies to midi note numbers and ansi note names."""
        from pygame.midi import frequency_to_midi, midi_to_frequency, midi_to_ansi_note
        self.assertEqual(frequency_to_midi(27.5), 21)
        self.assertEqual(frequency_to_midi(36.7), 26)
        self.assertEqual(frequency_to_midi(4186.0), 108)
        self.assertEqual(midi_to_frequency(21), 27.5)
        self.assertEqual(midi_to_frequency(26), 36.7)
        self.assertEqual(midi_to_frequency(108), 4186.0)
        self.assertEqual(midi_to_ansi_note(21), 'A0')
        self.assertEqual(midi_to_ansi_note(71), 'B4')
        self.assertEqual(midi_to_ansi_note(82), 'A#5')
        self.assertEqual(midi_to_ansi_note(83), 'B5')
        self.assertEqual(midi_to_ansi_note(93), 'A6')
        self.assertEqual(midi_to_ansi_note(94), 'A#6')
        self.assertEqual(midi_to_ansi_note(95), 'B6')
        self.assertEqual(midi_to_ansi_note(96), 'C7')
        self.assertEqual(midi_to_ansi_note(102), 'F#7')
        self.assertEqual(midi_to_ansi_note(108), 'C8')