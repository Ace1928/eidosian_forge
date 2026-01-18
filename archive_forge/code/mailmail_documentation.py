import email.utils
import getpass
import os
import sys
from configparser import ConfigParser
from io import StringIO
from twisted.copyright import version
from twisted.internet import reactor
from twisted.logger import Logger, textFileLogObserver
from twisted.mail import smtp


    @ivar allowUIDs: A list of UIDs which are allowed to send mail.
    @ivar allowGIDs: A list of GIDs which are allowed to send mail.
    @ivar denyUIDs: A list of UIDs which are not allowed to send mail.
    @ivar denyGIDs: A list of GIDs which are not allowed to send mail.

    @type defaultAccess: L{bool}
    @ivar defaultAccess: L{True} if access will be allowed when no other access
    control rule matches or L{False} if it will be denied in that case.

    @ivar useraccess: Either C{'allow'} to check C{allowUID} first
    or C{'deny'} to check C{denyUID} first.

    @ivar groupaccess: Either C{'allow'} to check C{allowGID} first or
    C{'deny'} to check C{denyGID} first.

    @ivar identities: A L{dict} mapping hostnames to credentials to use when
    sending mail to that host.

    @ivar smarthost: L{None} or a hostname through which all outgoing mail will
    be sent.

    @ivar domain: L{None} or the hostname with which to identify ourselves when
    connecting to an MTA.
    