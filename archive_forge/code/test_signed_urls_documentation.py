import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution

        Generate signed url from the Example Canned Policy in Amazon's
        documentation.
        