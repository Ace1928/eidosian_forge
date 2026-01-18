import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_sign_canned_policy_unicode(self):
    """
        Test signing the canned policy from amazon's cloudfront documentation.
        """
    expected = 'Nql641NHEUkUaXQHZINK1FZ~SYeUSoBJMxjdgqrzIdzV2gyEXPDNv0pYdWJkflDKJ3xIu7lbwRpSkG98NBlgPi4ZJpRRnVX4kXAJK6tdNx6FucDB7OVqzcxkxHsGFd8VCG1BkC-Afh9~lOCMIYHIaiOB6~5jt9w2EOwi6sIIqrg_'
    unicode_policy = six.text_type(self.canned_policy)
    sig = self.dist._sign_string(unicode_policy, private_key_string=self.pk_str)
    encoded_sig = self.dist._url_base64_encode(sig)
    self.assertEqual(expected, encoded_sig)