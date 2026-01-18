from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
In the event that 2 related items match our search criteria in this way:
    one item has an id that matches input
    one item has a name that matches input
    We should preference the id over the name.
    Otherwise, the universality of the controller_api lookup plugin is compromised.
    