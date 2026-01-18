import argparse
import datetime
import logging
import re
from oslo_serialization import jsonutils
from oslo_utils import strutils
from blazarclient import command
from blazarclient import exception
Delete a lease.