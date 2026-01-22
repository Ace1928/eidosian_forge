from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
class KeyVersionCompleter(ListCommandCompleter):

    def __init__(self, **kwargs):
        super(KeyVersionCompleter, self).__init__(collection=CRYPTO_KEY_VERSION_COLLECTION, list_command='kms keys versions list --uri', flags=['location', 'key', 'keyring'], **kwargs)