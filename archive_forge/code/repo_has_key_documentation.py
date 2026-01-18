from ...commands import Command
from ...controldir import ControlDir
Does a repo have a key?

    e.g.::

      bzr repo-has-key texts FILE-ID REVISION-ID
      bzr repo-has-key revisions REVISION-ID

    It either prints "True" or "False", and terminates with exit code 0 or 1
    respectively.
    