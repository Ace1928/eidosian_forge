import binascii
import re
from six import StringIO
from boto import storage_uri
from boto.exception import BotoClientError
from boto.gs.acl import SupportedPermissions as perms
from tests.integration.gs.testcase import GSTestCase
class GSStorageUriTest(GSTestCase):

    def testHasVersion(self):
        uri = storage_uri('gs://bucket/obj')
        self.assertFalse(uri.has_version())
        uri.version_id = 'versionid'
        self.assertTrue(uri.has_version())
        uri = storage_uri('gs://bucket/obj')
        uri.generation = 12345
        self.assertTrue(uri.has_version())
        uri.generation = None
        self.assertFalse(uri.has_version())
        uri = storage_uri('gs://bucket/obj')
        uri.generation = 0
        self.assertTrue(uri.has_version())

    def testCloneReplaceKey(self):
        b = self._MakeBucket()
        k = b.new_key('obj')
        k.set_contents_from_string('stringdata')
        orig_uri = storage_uri('gs://%s/' % b.name)
        uri = orig_uri.clone_replace_key(k)
        self.assertTrue(uri.has_version())
        self.assertRegexpMatches(str(uri.generation), '[0-9]+')

    def testSetAclXml(self):
        """Ensures that calls to the set_xml_acl functions succeed."""
        b = self._MakeBucket()
        k = b.new_key('obj')
        k.set_contents_from_string('stringdata')
        bucket_uri = storage_uri('gs://%s/' % b.name)
        bucket_uri.object_name = 'obj'
        bucket_acl = bucket_uri.get_acl()
        bucket_uri.object_name = None
        all_users_read_permission = "<Entry><Scope type='AllUsers'/><Permission>READ</Permission></Entry>"
        acl_string = re.sub('</Entries>', all_users_read_permission + '</Entries>', bucket_acl.to_xml())
        acl_no_owner_string = re.sub('<Owner>.*</Owner>', '', acl_string)
        bucket_uri.set_xml_acl(acl_string, 'obj')
        bucket_uri.set_xml_acl(acl_no_owner_string)
        bucket_uri.set_def_xml_acl(acl_no_owner_string)
        new_obj_acl_string = k.get_acl().to_xml()
        new_bucket_acl_string = bucket_uri.get_acl().to_xml()
        new_bucket_def_acl_string = bucket_uri.get_def_acl().to_xml()
        self.assertRegexpMatches(new_obj_acl_string, 'AllUsers')
        self.assertRegexpMatches(new_bucket_acl_string, 'AllUsers')
        self.assertRegexpMatches(new_bucket_def_acl_string, 'AllUsers')

    def testPropertiesUpdated(self):
        b = self._MakeBucket()
        bucket_uri = storage_uri('gs://%s' % b.name)
        key_uri = bucket_uri.clone_replace_name('obj')
        key_uri.set_contents_from_string('data1')
        self.assertRegexpMatches(str(key_uri.generation), '[0-9]+')
        k = b.get_key('obj')
        self.assertEqual(k.generation, key_uri.generation)
        self.assertEquals(k.get_contents_as_string(), 'data1')
        key_uri.set_contents_from_stream(StringIO.StringIO('data2'))
        self.assertRegexpMatches(str(key_uri.generation), '[0-9]+')
        self.assertGreater(key_uri.generation, k.generation)
        k = b.get_key('obj')
        self.assertEqual(k.generation, key_uri.generation)
        self.assertEquals(k.get_contents_as_string(), 'data2')
        key_uri.set_contents_from_file(StringIO.StringIO('data3'))
        self.assertRegexpMatches(str(key_uri.generation), '[0-9]+')
        self.assertGreater(key_uri.generation, k.generation)
        k = b.get_key('obj')
        self.assertEqual(k.generation, key_uri.generation)
        self.assertEquals(k.get_contents_as_string(), 'data3')

    def testCompose(self):
        data1 = 'hello '
        data2 = 'world!'
        expected_crc = 1238062967
        b = self._MakeBucket()
        bucket_uri = storage_uri('gs://%s' % b.name)
        key_uri1 = bucket_uri.clone_replace_name('component1')
        key_uri1.set_contents_from_string(data1)
        key_uri2 = bucket_uri.clone_replace_name('component2')
        key_uri2.set_contents_from_string(data2)
        key_uri_composite = bucket_uri.clone_replace_name('composite')
        components = [key_uri1, key_uri2]
        key_uri_composite.compose(components, content_type='text/plain')
        self.assertEquals(key_uri_composite.get_contents_as_string(), data1 + data2)
        composite_key = key_uri_composite.get_key()
        cloud_crc32c = binascii.hexlify(composite_key.cloud_hashes['crc32c'])
        self.assertEquals(cloud_crc32c, hex(expected_crc)[2:])
        self.assertEquals(composite_key.content_type, 'text/plain')
        key_uri1.bucket_name += '2'
        try:
            key_uri_composite.compose(components)
            self.fail("Composing between buckets didn't fail as expected.")
        except BotoClientError as err:
            self.assertEquals(err.reason, 'GCS does not support inter-bucket composing')