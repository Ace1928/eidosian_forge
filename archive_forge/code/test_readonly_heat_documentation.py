import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
Basic, read-only tests for Heat CLI client.

    Basic smoke test for the heat CLI commands which do not require
    creating or modifying stacks.
    