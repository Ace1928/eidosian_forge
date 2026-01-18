import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
def validate_cluster_args(positional_cluster, optional_cluster):
    if optional_cluster:
        print(CLUSTER_DEPRECATION_WARNING)
    if positional_cluster and optional_cluster:
        raise DuplicateArgs('<cluster>', (positional_cluster, optional_cluster))