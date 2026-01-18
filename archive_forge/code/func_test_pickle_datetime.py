import unittest
from six.moves import cPickle as pickle
import isodate
def test_pickle_datetime(self):
    """
        Parse an ISO datetime string and compare it to the expected value.
        """
    dti = isodate.parse_datetime('2012-10-26T09:33+00:00')
    for proto in range(0, pickle.HIGHEST_PROTOCOL + 1):
        pikl = pickle.dumps(dti, proto)
        self.assertEqual(dti, pickle.loads(pikl), 'pickle proto %d failed' % proto)