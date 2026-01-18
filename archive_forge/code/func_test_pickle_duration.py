import unittest
from six.moves import cPickle as pickle
import isodate
def test_pickle_duration(self):
    """
        Pickle / unpickle duration objects.
        """
    from isodate.duration import Duration
    dur = Duration()
    failed = []
    for proto in range(0, pickle.HIGHEST_PROTOCOL + 1):
        try:
            pikl = pickle.dumps(dur, proto)
            if dur != pickle.loads(pikl):
                raise Exception('not equal')
        except Exception as e:
            failed.append('pickle proto %d failed (%s)' % (proto, repr(e)))
    self.assertEqual(len(failed), 0, 'pickle protos failed: %s' % str(failed))