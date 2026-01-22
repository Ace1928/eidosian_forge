import argparse
import logging
import os
import sys
from keystoneauth1 import loading
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client
from novaclient import exceptions as exc
import novaclient.extension
from novaclient.i18n import _
from novaclient import utils
class NovaClientArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(NovaClientArgumentParser, self).__init__(*args, **kwargs)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.
        """
        self.print_usage(sys.stderr)
        choose_from = ' (choose from'
        progparts = self.prog.partition(' ')
        self.exit(2, _("error: %(errmsg)s\nTry '%(mainp)s help %(subp)s' for more information.\n") % {'errmsg': message.split(choose_from)[0], 'mainp': progparts[0], 'subp': progparts[2]})

    def _get_option_tuples(self, option_string):
        """returns (action, option, value) candidates for an option prefix

        Returns [first candidate] if all candidates refers to current and
        deprecated forms of the same options: "nova boot ... --key KEY"
        parsing succeed because --key could only match --key-name,
        --key_name which are current/deprecated forms of the same option.
        """
        option_tuples = super(NovaClientArgumentParser, self)._get_option_tuples(option_string)
        if len(option_tuples) > 1:
            normalizeds = [option.replace('_', '-') for action, option, value in option_tuples]
            if len(set(normalizeds)) == 1:
                return option_tuples[:1]
        return option_tuples