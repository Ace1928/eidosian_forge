import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_sign_custom_policy_1(self):
    """
        Test signing custom policy 1 from amazon's cloudfront documentation.
        """
    expected = 'cPFtRKvUfYNYmxek6ZNs6vgKEZP6G3Cb4cyVt~FjqbHOnMdxdT7eT6pYmhHYzuDsFH4Jpsctke2Ux6PCXcKxUcTIm8SO4b29~1QvhMl-CIojki3Hd3~Unxjw7Cpo1qRjtvrimW0DPZBZYHFZtiZXsaPt87yBP9GWnTQoaVysMxQ_'
    sig = self.dist._sign_string(self.custom_policy_1, private_key_string=self.pk_str)
    encoded_sig = self.dist._url_base64_encode(sig)
    self.assertEqual(expected, encoded_sig)