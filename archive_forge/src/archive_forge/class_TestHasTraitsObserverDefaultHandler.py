import unittest
from traits.api import (
from traits.observation.api import (
class TestHasTraitsObserverDefaultHandler(unittest.TestCase):
    """ Test the behaviour with dynamic default handler + container. """

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def test_default_not_called_if_init_contains_value(self):
        record = Record(number=123)
        self.assertEqual(record.default_call_count, 1)
        self.assertEqual(len(record.number_change_events), 1)
        event, = record.number_change_events
        self.assertEqual(event.object, record)
        self.assertEqual(event.name, 'number')
        self.assertEqual(event.old, 99)
        self.assertEqual(event.new, 123)

    def test_observe_extended_trait_in_list(self):
        album = Album()
        self.assertEqual(album.records_default_call_count, 0)
        self.assertEqual(len(album.record_number_change_events), 0)
        album.records[0].number += 1
        self.assertEqual(album.records_default_call_count, 1)
        self.assertEqual(len(album.record_number_change_events), 1)
        event, = album.record_number_change_events
        self.assertEqual(event.object, album.records[0])
        self.assertEqual(event.name, 'number')
        self.assertEqual(event.old, 99)
        self.assertEqual(event.new, 100)

    def test_observe_extended_trait_in_default_dict(self):
        album = Album()
        self.assertEqual(album.name_to_records_default_call_count, 0)
        self.assertEqual(len(album.name_to_records_clicked_events), 0)
        album.name_to_records['Record'].clicked = True
        self.assertEqual(len(album.name_to_records_clicked_events), 1)