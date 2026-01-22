from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
class StubbedBranchOpener(BranchOpener):
    """BranchOpener that provides canned answers.

        We implement the methods we need to to be able to control all the
        inputs to the `follow_reference` method, which is what is
        being tested in this class.
        """

    def __init__(self, references, policy):
        parent_cls = TestBranchOpenerCheckAndFollowBranchReference
        super(parent_cls.StubbedBranchOpener, self).__init__(policy)
        self._reference_values = {}
        for i in range(len(references) - 1):
            self._reference_values[references[i]] = references[i + 1]
        self.follow_reference_calls = []

    def follow_reference(self, url):
        self.follow_reference_calls.append(url)
        return self._reference_values[url]