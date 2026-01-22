from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('Tests use GS IAM model.')
@SkipForXML('XML IAM control is not supported.')
class TestIamCh(TestIamIntegration):
    """Integration tests for iam ch command."""

    def setUp(self):
        super(TestIamCh, self).setUp()
        self.bucket = self.CreateBucket()
        self.bucket2 = self.CreateBucket()
        self.object = self.CreateObject(bucket_uri=self.bucket, contents=b'foo')
        self.object2 = self.CreateObject(bucket_uri=self.bucket, contents=b'bar')
        self.bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        self.object2_iam_string = self.RunGsUtil(['iam', 'get', self.object2.uri], return_stdout=True)
        self.user = 'user:foo@bar.com'
        self.user2 = 'user:bar@foo.com'

    def test_patch_no_role(self):
        """Tests expected failure if no bindings are listed."""
        stderr = self.RunGsUtil(['iam', 'ch', self.bucket.uri], return_stderr=True, expected_status=1)
        self.assertIn('CommandException', stderr)

    def test_raises_error_message_for_d_flag_missing_argument(self):
        """Tests expected failure if no bindings are listed."""
        stderr = self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), '-d'], return_stderr=True, expected_status=1)
        self.assertIn('A -d flag is missing an argument specifying bindings to remove.', stderr)

    def test_path_mix_of_buckets_and_objects(self):
        """Tests expected failure if both buckets and objects are provided."""
        stderr = self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri, self.object.uri], return_stderr=True, expected_status=1)
        self.assertIn('CommandException', stderr)

    def test_path_file_url(self):
        """Tests expected failure is caught when a file url is provided."""
        stderr = self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), 'file://somefile'], return_stderr=True, expected_status=1)
        self.assertIn('AttributeError', stderr)

    def test_patch_single_grant_single_bucket(self):
        """Tests granting single role."""
        self.assertHasNo(self.bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)

    def test_patch_repeated_grant(self):
        """Granting multiple times for the same member will have no effect."""
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)

    def test_patch_single_remove_single_bucket(self):
        """Tests removing a single role."""
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        self.RunGsUtil(['iam', 'ch', '-d', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHasNo(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)

    def test_patch_null_remove(self):
        """Removing a non-existent binding will have no effect."""
        self.RunGsUtil(['iam', 'ch', '-d', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHasNo(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)
        self.assertEqualsPoliciesString(bucket_iam_string, self.bucket_iam_string)

    def test_patch_mixed_grant_remove_single_bucket(self):
        """Tests that mixing grant and remove requests will succeed."""
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user2, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), '-d', '%s:%s' % (self.user2, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)
        self.assertHasNo(bucket_iam_string, self.user2, IAM_BUCKET_READ_ROLE)

    def test_patch_public_grant_single_bucket(self):
        """Test public grant request interacts properly with existing members."""
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        self.RunGsUtil(['iam', 'ch', 'allUsers:%s' % IAM_BUCKET_READ_ROLE_ABBREV, self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, 'allUsers', IAM_BUCKET_READ_ROLE)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)

    def test_patch_remove_all_roles(self):
        """Remove with no roles specified will remove member from all bindings."""
        self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri])
        self.RunGsUtil(['iam', 'ch', '-d', self.user, self.bucket.uri])
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHasNo(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)

    def test_patch_single_object(self):
        """Tests object IAM patch behavior."""
        self.assertHasNo(self.object_iam_string, self.user, IAM_OBJECT_READ_ROLE)
        self.RunGsUtil(['iam', 'ch', '%s:legacyObjectReader' % self.user, self.object.uri])
        object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        self.assertHas(object_iam_string, self.user, IAM_OBJECT_READ_ROLE)

    def test_patch_multithreaded_single_object(self):
        """Tests the edge-case behavior of multithreaded execution."""
        self.assertHasNo(self.object_iam_string, self.user, IAM_OBJECT_READ_ROLE)
        self.RunGsUtil(['-m', 'iam', 'ch', '%s:legacyObjectReader' % self.user, self.object.uri])
        object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        self.assertHas(object_iam_string, self.user, IAM_OBJECT_READ_ROLE)

    def test_patch_invalid_input(self):
        """Tests that listing bindings after a bucket will throw an error."""
        stderr = self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri, '%s:%s' % (self.user2, IAM_BUCKET_READ_ROLE_ABBREV)], return_stderr=True, expected_status=1)
        self.assertIn('CommandException', stderr)
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)
        self.assertHasNo(bucket_iam_string, self.user2, IAM_BUCKET_READ_ROLE)

    def test_patch_disallowed_binding_type(self):
        """Tests that we disallow certain binding types with appropriate err."""
        stderr = self.RunGsUtil(['iam', 'ch', 'projectOwner:my-project:admin', self.bucket.uri], return_stderr=True, expected_status=1)
        self.assertIn('not supported', stderr)

    def test_patch_remove_disallowed_binding_type(self):
        """Tests that we can remove project convenience values."""
        disallowed_member = 'projectViewer:%s' % PopulateProjectId()
        policy_file_path = self.CreateTempFile(contents=json.dumps(patch_binding(json.loads(self.bucket_iam_string), IAM_OBJECT_READ_ROLE, gen_binding(IAM_OBJECT_READ_ROLE, members=[disallowed_member]))).encode(UTF8))
        self.RunGsUtil(['iam', 'set', policy_file_path, self.bucket.uri])
        iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHas(iam_string, disallowed_member, IAM_OBJECT_READ_ROLE)
        self.RunGsUtil(['iam', 'ch', '-d', disallowed_member, self.bucket.uri])
        iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        self.assertHasNo(iam_string, disallowed_member, IAM_OBJECT_READ_ROLE)

    def test_patch_multiple_objects(self):
        """Tests IAM patch against multiple objects."""
        self.RunGsUtil(['iam', 'ch', '-r', '%s:legacyObjectReader' % self.user, self.bucket.uri])
        object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        object2_iam_string = self.RunGsUtil(['iam', 'get', self.object2.uri], return_stdout=True)
        self.assertHas(object_iam_string, self.user, IAM_OBJECT_READ_ROLE)
        self.assertHas(object2_iam_string, self.user, IAM_OBJECT_READ_ROLE)

    def test_patch_multithreaded_multiple_objects(self):
        """Tests multithreaded behavior against multiple objects."""
        self.RunGsUtil(['-m', 'iam', 'ch', '-r', '%s:legacyObjectReader' % self.user, self.bucket.uri])
        object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        object2_iam_string = self.RunGsUtil(['iam', 'get', self.object2.uri], return_stdout=True)
        self.assertHas(object_iam_string, self.user, IAM_OBJECT_READ_ROLE)
        self.assertHas(object2_iam_string, self.user, IAM_OBJECT_READ_ROLE)

    def test_patch_error(self):
        """See TestIamSet.test_set_error."""
        stderr = self.RunGsUtil(['iam', 'ch', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri, 'gs://%s' % self.nonexistent_bucket_name, self.bucket2.uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('not found: 404.', stderr)
        else:
            self.assertIn('BucketNotFoundException', stderr)
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        bucket2_iam_string = self.RunGsUtil(['iam', 'get', self.bucket2.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)
        self.assertEqualsPoliciesString(bucket2_iam_string, self.bucket_iam_string)

    def test_patch_force_error(self):
        """See TestIamSet.test_set_force_error."""
        stderr = self.RunGsUtil(['iam', 'ch', '-f', '%s:%s' % (self.user, IAM_BUCKET_READ_ROLE_ABBREV), self.bucket.uri, 'gs://%s' % self.nonexistent_bucket_name, self.bucket2.uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('not found: 404.', stderr)
        else:
            self.assertIn('CommandException', stderr)
        bucket_iam_string = self.RunGsUtil(['iam', 'get', self.bucket.uri], return_stdout=True)
        bucket2_iam_string = self.RunGsUtil(['iam', 'get', self.bucket2.uri], return_stdout=True)
        self.assertHas(bucket_iam_string, self.user, IAM_BUCKET_READ_ROLE)
        self.assertHas(bucket2_iam_string, self.user, IAM_BUCKET_READ_ROLE)

    def test_patch_multithreaded_error(self):
        """See TestIamSet.test_set_multithreaded_error."""

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stderr = self.RunGsUtil(['-m', 'iam', 'ch', '-r', '%s:legacyObjectReader' % self.user, 'gs://%s' % self.nonexistent_bucket_name, self.bucket.uri], return_stderr=True, expected_status=1)
            if self._use_gcloud_storage:
                self.assertIn('not found: 404.', stderr)
            else:
                self.assertIn('BucketNotFoundException', stderr)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check2():
            object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
            object2_iam_string = self.RunGsUtil(['iam', 'get', self.object2.uri], return_stdout=True)
            self.assertEqualsPoliciesString(self.object_iam_string, object_iam_string)
            self.assertEqualsPoliciesString(self.object_iam_string, object2_iam_string)
        _Check1()
        _Check2()

    def test_assert_has(self):
        test_policy = {'bindings': [{'members': ['allUsers'], 'role': 'roles/storage.admin'}, {'members': ['user:foo@bar.com', 'serviceAccount:bar@foo.com'], 'role': IAM_BUCKET_READ_ROLE}]}
        self.assertHas(json.dumps(test_policy), 'allUsers', 'roles/storage.admin')
        self.assertHas(json.dumps(test_policy), 'user:foo@bar.com', IAM_BUCKET_READ_ROLE)
        self.assertHasNo(json.dumps(test_policy), 'allUsers', IAM_BUCKET_READ_ROLE)
        self.assertHasNo(json.dumps(test_policy), 'user:foo@bar.com', 'roles/storage.admin')

    def assertHas(self, policy, member, role):
        """Asserts a member has permission for role.

    Given an IAM policy, check if the specified member is bound to the
    specified role. Does not check group inheritence -- that is, if checking
    against the [{'member': ['allUsers'], 'role': X}] policy, this function
    will still raise an exception when testing for any member other than
    'allUsers' against role X.

    This function does not invoke the TestIamPolicy endpoints to smartly check
    IAM policy resolution. This function is simply to assert the expected IAM
    policy is returned, not whether or not the IAM policy is being invoked as
    expected.

    Args:
      policy: Policy object as formatted by IamCommand._GetIam()
      member: A member string (e.g. 'user:foo@bar.com').
      role: A fully specified role (e.g. 'roles/storage.admin')

    Raises:
      AssertionError if member is not bound to role.
    """
        policy = json.loads(policy)
        bindings = dict(((p['role'], p) for p in policy.get('bindings', [])))
        if role in bindings:
            if member in bindings[role]['members']:
                return
        raise AssertionError("Member '%s' does not have permission '%s' in policy %s" % (member, role, policy))

    def assertHasNo(self, policy, member, role):
        """Functions as logical compliment of TestIamCh.assertHas()."""
        try:
            self.assertHas(policy, member, role)
        except AssertionError:
            pass
        else:
            raise AssertionError("Member '%s' has permission '%s' in policy %s" % (member, role, policy))