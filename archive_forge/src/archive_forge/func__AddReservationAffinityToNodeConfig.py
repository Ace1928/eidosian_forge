from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def _AddReservationAffinityToNodeConfig(node_config, options, messages):
    """Adds ReservationAffinity to NodeConfig."""
    affinity = options.reservation_affinity
    if options.reservation and affinity != 'specific':
        raise util.Error(RESERVATION_AFFINITY_NON_SPECIFIC_WITH_RESERVATION_NAME_ERROR_MSG.format(affinity=affinity))
    if not options.reservation and affinity == 'specific':
        raise util.Error(RESERVATION_AFFINITY_SPECIFIC_WITHOUT_RESERVATION_NAME_ERROR_MSG)
    if affinity == 'none':
        node_config.reservationAffinity = messages.ReservationAffinity(consumeReservationType=messages.ReservationAffinity.ConsumeReservationTypeValueValuesEnum.NO_RESERVATION)
    elif affinity == 'any':
        node_config.reservationAffinity = messages.ReservationAffinity(consumeReservationType=messages.ReservationAffinity.ConsumeReservationTypeValueValuesEnum.ANY_RESERVATION)
    elif affinity == 'specific':
        node_config.reservationAffinity = messages.ReservationAffinity(consumeReservationType=messages.ReservationAffinity.ConsumeReservationTypeValueValuesEnum.SPECIFIC_RESERVATION, key='compute.googleapis.com/reservation-name', values=[options.reservation])