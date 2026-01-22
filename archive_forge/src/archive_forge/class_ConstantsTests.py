import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
@skipIf(not cryptography, 'Cannot run without cryptography')
class ConstantsTests(TestCase):
    """
    Tests for the constants used by the SFTP protocol implementation.

    @ivar filexferSpecExcerpts: Excerpts from the
        draft-ietf-secsh-filexfer-02.txt (draft) specification of the SFTP
        protocol.  There are more recent drafts of the specification, but this
        one describes version 3, which is what conch (and OpenSSH) implements.
    """
    filexferSpecExcerpts = ["\n           The following values are defined for packet types.\n\n                #define SSH_FXP_INIT                1\n                #define SSH_FXP_VERSION             2\n                #define SSH_FXP_OPEN                3\n                #define SSH_FXP_CLOSE               4\n                #define SSH_FXP_READ                5\n                #define SSH_FXP_WRITE               6\n                #define SSH_FXP_LSTAT               7\n                #define SSH_FXP_FSTAT               8\n                #define SSH_FXP_SETSTAT             9\n                #define SSH_FXP_FSETSTAT           10\n                #define SSH_FXP_OPENDIR            11\n                #define SSH_FXP_READDIR            12\n                #define SSH_FXP_REMOVE             13\n                #define SSH_FXP_MKDIR              14\n                #define SSH_FXP_RMDIR              15\n                #define SSH_FXP_REALPATH           16\n                #define SSH_FXP_STAT               17\n                #define SSH_FXP_RENAME             18\n                #define SSH_FXP_READLINK           19\n                #define SSH_FXP_SYMLINK            20\n                #define SSH_FXP_STATUS            101\n                #define SSH_FXP_HANDLE            102\n                #define SSH_FXP_DATA              103\n                #define SSH_FXP_NAME              104\n                #define SSH_FXP_ATTRS             105\n                #define SSH_FXP_EXTENDED          200\n                #define SSH_FXP_EXTENDED_REPLY    201\n\n           Additional packet types should only be defined if the protocol\n           version number (see Section ``Protocol Initialization'') is\n           incremented, and their use MUST be negotiated using the version\n           number.  However, the SSH_FXP_EXTENDED and SSH_FXP_EXTENDED_REPLY\n           packets can be used to implement vendor-specific extensions.  See\n           Section ``Vendor-Specific-Extensions'' for more details.\n        ", '\n            The flags bits are defined to have the following values:\n\n                #define SSH_FILEXFER_ATTR_SIZE          0x00000001\n                #define SSH_FILEXFER_ATTR_UIDGID        0x00000002\n                #define SSH_FILEXFER_ATTR_PERMISSIONS   0x00000004\n                #define SSH_FILEXFER_ATTR_ACMODTIME     0x00000008\n                #define SSH_FILEXFER_ATTR_EXTENDED      0x80000000\n\n        ', "\n            The `pflags' field is a bitmask.  The following bits have been\n           defined.\n\n                #define SSH_FXF_READ            0x00000001\n                #define SSH_FXF_WRITE           0x00000002\n                #define SSH_FXF_APPEND          0x00000004\n                #define SSH_FXF_CREAT           0x00000008\n                #define SSH_FXF_TRUNC           0x00000010\n                #define SSH_FXF_EXCL            0x00000020\n        ", '\n            Currently, the following values are defined (other values may be\n           defined by future versions of this protocol):\n\n                #define SSH_FX_OK                            0\n                #define SSH_FX_EOF                           1\n                #define SSH_FX_NO_SUCH_FILE                  2\n                #define SSH_FX_PERMISSION_DENIED             3\n                #define SSH_FX_FAILURE                       4\n                #define SSH_FX_BAD_MESSAGE                   5\n                #define SSH_FX_NO_CONNECTION                 6\n                #define SSH_FX_CONNECTION_LOST               7\n                #define SSH_FX_OP_UNSUPPORTED                8\n        ']

    def test_constantsAgainstSpec(self):
        """
        The constants used by the SFTP protocol implementation match those
        found by searching through the spec.
        """
        constants = {}
        for excerpt in self.filexferSpecExcerpts:
            for line in excerpt.splitlines():
                m = re.match('^\\s*#define SSH_([A-Z_]+)\\s+([0-9x]*)\\s*$', line)
                if m:
                    constants[m.group(1)] = int(m.group(2), 0)
        self.assertTrue(len(constants) > 0, 'No constants found (the test must be buggy).')
        for k, v in constants.items():
            self.assertEqual(v, getattr(filetransfer, k))