from breezy import errors
from breezy.bzr import knit
from breezy.tests.per_repository_reference import \
def test_unordered_fetch_complex_split(self):
    self.make_complex_split()
    keys = [(b'f-id', bytes([r])) for r in bytearray(b'ABCDEG')]
    self.stacked_repo.lock_read()
    self.addCleanup(self.stacked_repo.unlock)
    stream = self.stacked_repo.texts.get_record_stream(keys, 'unordered', False)
    record_keys = set()
    for record in stream:
        if record.storage_kind == 'absent':
            raise ValueError('absent record: {}'.format(record.key))
        record_keys.add(record.key)
    self.assertEqual(keys, sorted(record_keys))