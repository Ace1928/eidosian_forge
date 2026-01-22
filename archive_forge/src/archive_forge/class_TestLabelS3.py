from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import xml
from xml.dom.minidom import parseString
from xml.sax import _exceptions as SaxExceptions
import six
import boto
from boto import handler
from boto.s3.tagging import Tags
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
@SkipForGS('Tests use S3-style XML passthrough.')
class TestLabelS3(testcase.GsUtilIntegrationTestCase):
    """S3-specific tests. Most other test cases are covered in TestLabelGS."""
    _label_xml = parseString('<Tagging><TagSet>' + '<Tag><Key>' + KEY1 + '</Key><Value>' + VALUE1 + '</Value></Tag>' + '<Tag><Key>' + KEY2 + '</Key><Value>' + VALUE2 + '</Value></Tag>' + '</TagSet></Tagging>').toprettyxml(indent='    ')

    def setUp(self):
        super(TestLabelS3, self).setUp()
        self.xml_fpath = self.CreateTempFile(contents=self._label_xml.encode(UTF8))

    def DoAssertItemsMatch(self, item1, item2):
        if six.PY2:
            self.assertItemsEqual(item1, item2)
        else:
            self.assertCountEqual(item1, item2)

    def _LabelDictFromXmlString(self, xml_str):
        label_dict = {}
        tags_list = Tags()
        h = handler.XmlHandler(tags_list, None)
        try:
            xml.sax.parseString(xml_str, h)
        except SaxExceptions.SAXParseException as e:
            raise CommandException('Requested labels/tagging config is invalid: %s at line %s, column %s' % (e.getMessage(), e.getLineNumber(), e.getColumnNumber()))
        for tagset_list in tags_list:
            for tag in tagset_list:
                label_dict[tag.key] = tag.value
        return label_dict

    def testSetAndGet(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['label', 'set', self.xml_fpath, suri(bucket_uri)], return_stderr=True)
        expected_output = _get_label_setting_output(self._use_gcloud_storage, suri(bucket_uri))
        if self._use_gcloud_storage:
            self.assertIn(expected_output, stderr)
        else:
            self.assertEqual(stderr.strip(), expected_output)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
            self.DoAssertItemsMatch(self._LabelDictFromXmlString(stdout), self._LabelDictFromXmlString(self._label_xml))
        _Check1()

    def testCh(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['label', 'ch', '-l', '%s:%s' % (KEY1, VALUE1), '-l', '%s:%s' % (KEY2, VALUE2), suri(bucket_uri)])

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
            self.DoAssertItemsMatch(self._LabelDictFromXmlString(stdout), self._LabelDictFromXmlString(self._label_xml))
        _Check1()
        self.RunGsUtil(['label', 'ch', '-d', KEY1, '-l', 'new_key:new_value', '-d', 'nonexistent-key', suri(bucket_uri)])
        expected_dict = {KEY2: VALUE2, 'new_key': 'new_value'}

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
            self.DoAssertItemsMatch(self._LabelDictFromXmlString(stdout), expected_dict)
        _Check2()