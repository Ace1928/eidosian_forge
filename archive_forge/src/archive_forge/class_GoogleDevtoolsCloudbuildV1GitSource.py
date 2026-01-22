from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1GitSource(_messages.Message):
    """Location of the source in any accessible Git repository.

  Fields:
    dir: Directory, relative to the source root, in which to run the build.
      This must be a relative path. If a step's `dir` is specified and is an
      absolute path, this value is ignored for that step's execution.
    revision: The revision to fetch from the Git repository such as a branch,
      a tag, a commit SHA, or any Git ref. Cloud Build uses `git fetch` to
      fetch the revision from the Git repository; therefore make sure that the
      string you provide for `revision` is parsable by the command. For
      information on string values accepted by `git fetch`, see https://git-
      scm.com/docs/gitrevisions#_specifying_revisions. For information on `git
      fetch`, see https://git-scm.com/docs/git-fetch.
    url: Location of the Git repo to build. This will be used as a `git
      remote`, see https://git-scm.com/docs/git-remote.
  """
    dir = _messages.StringField(1)
    revision = _messages.StringField(2)
    url = _messages.StringField(3)