import logging
from osc_lib.command import command
from openstackclient.i18n import _
Return prior and implied role id(s)

    If prior and implied role id(s) are retrievable from identity
    client, return tuple containing them.
    