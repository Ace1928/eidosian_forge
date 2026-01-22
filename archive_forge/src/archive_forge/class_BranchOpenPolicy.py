import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class BranchOpenPolicy:
    """Policy on how to open branches.

    In particular, a policy determines which branches are okay to open by
    checking their URLs and deciding whether or not to follow branch
    references.
    """

    def should_follow_references(self):
        """Whether we traverse references when mirroring.

        Subclasses must override this method.

        If we encounter a branch reference and this returns false, an error is
        raised.

        :returns: A boolean to indicate whether to follow a branch reference.
        """
        raise NotImplementedError(self.should_follow_references)

    def transform_fallback_location(self, branch, url):
        """Validate, maybe modify, 'url' to be used as a stacked-on location.

        :param branch:  The branch that is being opened.
        :param url: The URL that the branch provides for its stacked-on
            location.
        :return: (new_url, check) where 'new_url' is the URL of the branch to
            actually open and 'check' is true if 'new_url' needs to be
            validated by check_and_follow_branch_reference.
        """
        raise NotImplementedError(self.transform_fallback_location)

    def check_one_url(self, url):
        """Check a URL.

        Subclasses must override this method.

        :param url: The source URL to check.
        :raise BadUrl: subclasses are expected to raise this or a subclass
            when it finds a URL it deems to be unacceptable.
        """
        raise NotImplementedError(self.check_one_url)