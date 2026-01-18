import os
from .file import GitFile
from .index import commit_tree, iter_fresh_objects
from .reflog import drop_reflog_entry, read_reflog
Create a new stash.

        Args:
          committer: Optional committer name to use
          author: Optional author name to use
          message: Optional commit message
        