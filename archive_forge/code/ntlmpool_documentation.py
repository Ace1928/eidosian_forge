from __future__ import absolute_import
import warnings
from logging import getLogger
from ntlm import ntlm
from .. import HTTPSConnectionPool
from ..packages.six.moves.http_client import HTTPSConnection

        authurl is a random URL on the server that is protected by NTLM.
        user is the Windows user, probably in the DOMAIN\username format.
        pw is the password for the user.
        