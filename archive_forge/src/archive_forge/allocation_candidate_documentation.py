import argparse
import collections
from osc_lib.command import command
from osc_lib import exceptions
from osc_placement.resources import common
from osc_placement import version
List allocation candidates.

    Returns a representation of a collection of allocation requests and
    resource provider summaries. Each allocation request has information
    to issue an ``openstack resource provider allocation set`` request to claim
    resources against a related set of resource providers.

    As several allocation requests are available its necessary to select one.
    To make a decision, resource provider summaries are provided with the
    inventory/capacity information.

    For example::

      $ export OS_PLACEMENT_API_VERSION=1.10
      $ openstack allocation candidate list --resource VCPU=1
      +---+------------+-------------------------+-------------------------+
      | # | allocation | resource provider       | inventory used/capacity |
      +---+------------+-------------------------+-------------------------+
      | 1 | VCPU=1     | 66bcaca9-9263-45b1-a569 | VCPU=0/128              |
      |   |            | -ea708ff7a968           |                         |
      +---+------------+-------------------------+-------------------------+

    In this case, the user is looking for resource providers that can have
    capacity to allocate 1 ``VCPU`` resource class. There is one resource
    provider that can serve that allocation request and that resource providers
    current ``VCPU`` inventory used is 0 and available capacity is 128.

    This command requires at least ``--os-placement-api-version 1.10``.
    