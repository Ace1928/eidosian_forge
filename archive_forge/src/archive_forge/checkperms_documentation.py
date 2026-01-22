import os

    Represents a set of specifications for permissions.

    Typically reads from a file that looks like this::

      rwxrwxrwx user:group filename

    If the filename ends in /, then it expected to be a directory, and
    the directory is made executable automatically, and the contents
    of the directory are given the same permission (recursively).  By
    default the executable bit on files is left as-is, unless the
    permissions specifically say it should be on in some way.

    You can use 'nomodify filename' for permissions to say that any
    permission is okay, and permissions should not be changed.

    Use 'noexist filename' to say that a specific file should not
    exist.

    Use 'symlink filename symlinked_to' to assert a symlink destination

    The entire file is read, and most specific rules are used for each
    file (i.e., a rule for a subdirectory overrides the rule for a
    superdirectory).  Order does not matter.
    